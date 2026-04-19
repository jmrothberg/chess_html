"""
Minimal stdlib HTTP server that bridges chess.html (browser) to Chess_Inference
(PyTorch). Serves chess.html at / and exposes:

  GET  /api/models           -> {"models": ["file.pth", ...]}
  POST /api/predict_move     -> body {"history": "e2e4 e7e5 ...",
                                      "model":   "file.pth",
                                      "top_k":   10}
                                returns {"moves": ["g1f3", "b1c3", ...]}

Run:
    python3 "chess/chess_server.py"
Then open http://localhost:5858/chess.html
"""

import importlib.util
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "Chess_LLM_models copy")
HOST, PORT = "127.0.0.1", 5858


def _load_inference_module():
    path = os.path.join(HERE, "Chess_Inference copy.py")
    spec = importlib.util.spec_from_file_location("chess_inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chess_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


inference = _load_inference_module()

_models = {}
_models_lock = threading.Lock()


def list_model_files():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pth"))


def get_model(filename):
    with _models_lock:
        if filename in _models:
            return _models[filename]
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(filename)
        print(f"[chess_server] Loading {filename} ...", flush=True)
        model, *_meta, tokenizer = inference.load_model_file(checkpoint_path=path)
        if model is None:
            raise RuntimeError(f"load_model_file returned None for {filename}")
        reverse = {v: k for k, v in tokenizer.items()} if tokenizer else None
        entry = (model, tokenizer, reverse)
        _models[filename] = entry
        print(f"[chess_server] Loaded {filename}", flush=True)
        return entry


def _tokenize_4token(input_text):
    """Turn '<STARTGAME> e2e4 e7e5 ...' into the 4-token-per-ply id sequence
    expected by ChessModel(token_mode='4token'). Mirrors the tokenizer used in
    Chess_Inference.generate_response but without the broken model-output code.
    Returns (tokens, ply).
    """
    FROM_OFFSET = inference.FROM_OFFSET
    TO_OFFSET = inference.TO_OFFSET
    COLOR_OFFSET = inference.COLOR_OFFSET
    STARTGAME = inference.STARTGAME
    EOFG = inference.EOFG
    W_RESULT = inference.W_RESULT
    D_RESULT = inference.D_RESULT

    tokens = []
    ply = 0
    i = 0
    n = len(input_text)
    while i < n:
        if input_text[i:i + 11] == '<STARTGAME>':
            tokens.append(STARTGAME); ply = 0; i += 11
        elif input_text[i:i + 6] == '<EOFG>':
            tokens.append(EOFG); i += 6
        elif input_text[i:i + 3] == '<W>':
            tokens.append(W_RESULT); i += 3
        elif input_text[i:i + 3] == '<D>':
            tokens.append(D_RESULT); i += 3
        elif input_text[i].isspace():
            i += 1
        elif i + 4 <= n:
            move_str = None
            if i + 5 <= n and input_text[i + 4].isalpha() and input_text[i + 4].lower() in 'qrbn':
                c = input_text[i:i + 5]
                if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                    move_str = c; i += 5
            if move_str is None:
                c = input_text[i:i + 4]
                if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                    move_str = c; i += 4
                else:
                    i += 1; continue
            is_white = (ply % 2 == 0)
            ct, ft, tt, pt = inference.parse_uci_move(move_str, is_white)
            tokens.extend([ct, ft, tt, pt])
            ply += 1
        else:
            i += 1
    return tokens, ply


def _predict_4token(model, input_text, top_k=10, device_str='cpu'):
    """Drop-in replacement for inference.generate_response 4-token path.
    Calls head_from / head_to / head_promo on the raw hidden state returned by
    ChessModel.forward() in 4token mode.
    """
    device = torch.device(device_str)
    model.eval()
    model.to(device)

    tokens, ply = _tokenize_4token(input_text)
    block_size = model.block_size if hasattr(model, 'block_size') else 512
    if len(tokens) > block_size:
        tokens = tokens[-block_size:]

    input_seq = torch.tensor([tokens], dtype=torch.long, device=device)
    model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    FROM_OFFSET = inference.FROM_OFFSET
    TO_OFFSET = inference.TO_OFFSET
    COLOR_OFFSET = inference.COLOR_OFFSET

    with torch.no_grad():
        is_white = (ply % 2 == 0)
        color_tok = COLOR_OFFSET + (0 if is_white else 1)
        input_seq = torch.cat([input_seq, torch.tensor([[color_tok]], device=device)], dim=1)
        if input_seq.shape[1] > block_size:
            input_seq = input_seq[:, -block_size:]

        hidden, _ = model(input_seq)               # (1, T, C)
        from_logits = model_raw.head_from(hidden[0, -1])  # (64,)
        from_probs = F.softmax(from_logits, dim=-1)

        num_from = min(top_k, 64)
        top_from_probs, top_from_sqs = torch.topk(from_probs, k=num_from)

        candidates = []  # (score, from_sq, to_sq, promo_idx)
        for fi in range(num_from):
            from_sq = top_from_sqs[fi].item()
            from_prob = top_from_probs[fi].item()
            from_tok = FROM_OFFSET + from_sq

            seq_f = torch.cat([input_seq, torch.tensor([[from_tok]], device=device)], dim=1)
            if seq_f.shape[1] > block_size:
                seq_f = seq_f[:, -block_size:]

            hidden2, _ = model(seq_f)
            h_last = hidden2[0, -1]
            from_emb = model_raw.emb_from(torch.tensor(from_sq, device=device))
            h_cond = h_last + from_emb
            to_logits = model_raw.head_to(h_cond)
            to_logits[from_sq] = float('-inf')
            to_probs = F.softmax(to_logits, dim=-1)

            num_to = min(3, 64)
            top_to_probs, top_to_sqs = torch.topk(to_probs, k=num_to)

            for ti in range(num_to):
                to_sq = top_to_sqs[ti].item()
                to_prob = top_to_probs[ti].item()
                score = from_prob * to_prob

                from_rank = 8 - (from_sq // 8)
                to_rank = 8 - (to_sq // 8)
                is_promo = (
                    (is_white and from_rank == 7 and to_rank == 8)
                    or (not is_white and from_rank == 2 and to_rank == 1)
                )
                if is_promo:
                    to_tok = TO_OFFSET + to_sq
                    seq_t = torch.cat([seq_f, torch.tensor([[to_tok]], device=device)], dim=1)
                    if seq_t.shape[1] > block_size:
                        seq_t = seq_t[:, -block_size:]
                    hidden3, _ = model(seq_t)
                    promo_logits = model_raw.head_promo(hidden3[0, -1])
                    promo_idx = promo_logits.argmax(dim=-1).item()
                    if promo_idx == 0:
                        promo_idx = 1
                else:
                    promo_idx = 0
                candidates.append((score, from_sq, to_sq, promo_idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        promo_chars = ['', 'q', 'r', 'b', 'n']
        seen = set()
        result = []
        for _, fr, to, pr in candidates:
            uci = inference.square_to_uci(fr) + inference.square_to_uci(to)
            if 0 < pr < len(promo_chars):
                uci += promo_chars[pr]
            if uci in seen:
                continue
            seen.add(uci)
            result.append(uci)
            if len(result) >= top_k:
                break
        return result


def generate_moves(model, tokenizer, reverse, input_text, top_k=10):
    token_mode = getattr(model, '_token_mode', None)
    if token_mode is None:
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        token_mode = getattr(raw, 'token_mode', '4token')
    if token_mode == 'classic':
        return inference._generate_classic(model, tokenizer, reverse, input_text, top_k)
    return _predict_4token(model, input_text, top_k=top_k)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[chess_server] " + (fmt % args) + "\n")

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/models":
            self._send_json(200, {"models": list_model_files()})
            return

        rel = self.path.lstrip("/") or "chess.html"
        if ".." in rel.split("/"):
            self._send_json(403, {"error": "forbidden"})
            return
        full = os.path.join(HERE, rel)
        if not os.path.isfile(full):
            self._send_json(404, {"error": f"not found: {rel}"})
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
        self._send_file(full, ctype)

    def do_POST(self):
        if self.path != "/api/predict_move":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"bad json: {e}"})
            return

        history = (req.get("history") or "").strip()
        model_name = req.get("model")
        top_k = int(req.get("top_k") or 10)
        if not model_name:
            self._send_json(400, {"error": "missing 'model'"})
            return

        try:
            model, tokenizer, reverse = get_model(model_name)
        except FileNotFoundError:
            self._send_json(404, {"error": f"model not found: {model_name}"})
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"load failed: {e}"})
            return

        input_text = "<STARTGAME> " + history if history else "<STARTGAME>"
        try:
            moves = generate_moves(model, tokenizer, reverse, input_text, top_k=top_k)
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"predict failed: {e}"})
            return

        self._send_json(200, {"moves": moves or []})


def main():
    models = list_model_files()
    print(f"[chess_server] Models dir: {MODELS_DIR}")
    if models:
        print(f"[chess_server] Found {len(models)} model(s):")
        for m in models:
            print(f"  - {m}")
    else:
        print("[chess_server] WARNING: no .pth files found in models dir")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[chess_server] Open http://{HOST}:{PORT}/chess.html")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[chess_server] shutting down")
        srv.server_close()


if __name__ == "__main__":
    main()

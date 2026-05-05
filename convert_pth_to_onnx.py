#!/usr/bin/env python3
"""Convert a ChessModel .pth checkpoint to a browser-loadable .onnx file.

Run once per checkpoint. The resulting .onnx is opened directly by
chess_full.html via a file picker — no Python is needed at runtime.

  python3 chess/convert_pth_to_onnx.py \
      --pth  "chess/Chess_LLM_models copy/<file>.pth" \
      --out  "chess/Chess_LLM_models copy/<file>.onnx"

By default the weights are converted to fp16 (~halves file size, faster on
WebGPU). Pass --no-fp16 to keep fp32, or --int8 for dynamic int8 quantization
(smallest, slight quality loss).
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_inference():
    path = os.path.join(HERE, "Chess_Inference copy.py")
    spec = importlib.util.spec_from_file_location("chess_inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chess_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


inference = _load_inference()


class FourTokenExport(nn.Module):
    """Wrap a 4-token ChessModel so one forward returns from/to/promo logits.

    `from_idx` is an extra input so the same graph serves both the FROM step
    (where only `from_logits` matters and `from_idx` is a dummy) and the per-FROM
    TO step (where `from_idx` is the just-chosen square and `to_logits` matters).
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, from_idx):
        hidden, _ = self.m(input_ids)              # (1, T, C)
        last = hidden[:, -1]                       # (1, C)
        from_logits = self.m.head_from(last)       # (1, 64)
        promo_logits = self.m.head_promo(last)     # (1, 5)
        from_emb = self.m.emb_from(from_idx)       # (1, C)
        to_logits = self.m.head_to(last + from_emb)
        return from_logits, to_logits, promo_logits


class ClassicExport(nn.Module):
    """Wrap a classic 1-token ChessModel; return logits at the last position."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        logits, _ = self.m(input_ids)              # (1, T, V)
        return logits[:, -1]                       # (1, V)


def detect_n_kv_heads(model):
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        return None
    block = model.blocks[0]
    attn = getattr(block, "attn", None)
    if attn is not None and hasattr(attn, "n_kv_heads"):
        return int(attn.n_kv_heads)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to source .pth checkpoint")
    ap.add_argument("--out", required=True, help="Path to write .onnx")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                    help="Convert weights to float16 after export (default)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="Keep weights as float32")
    ap.add_argument("--int8", action="store_true",
                    help="Apply dynamic int8 quantization (overrides --fp16)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed-len", type=int, default=16,
                    help="Sample sequence length used for tracing")
    args = ap.parse_args()

    if not os.path.isfile(args.pth):
        print(f"ERROR: not a file: {args.pth}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {args.pth} ...")
    model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer = \
        inference.load_model_file(checkpoint_path=args.pth)
    if model is None:
        print("ERROR: load_model_file returned None.", file=sys.stderr)
        sys.exit(3)

    token_mode = getattr(model, "_token_mode", "4token")
    n_kv_heads = detect_n_kv_heads(model)

    model.eval()
    # Drop game-mask path: the browser loop is always single-game continuation.
    if hasattr(model, "use_chess"):
        model.use_chess = False
    if hasattr(model, "start_game_token"):
        model.start_game_token = None

    # Pre-convert causal_mask buffers to bool so the .bool() cast inside
    # MultiQueryAttention becomes a no-op and onnxconverter-common's float16
    # pass doesn't leave a fp32-typed Cast node that mismatches its fp16 data.
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "causal_mask"):
                attn.causal_mask = attn.causal_mask.bool()

    T = max(2, args.seed_len)
    if token_mode == "4token":
        wrapper = FourTokenExport(model)
        seed_ids = torch.zeros(1, T, dtype=torch.long)
        seed_ids[0, 0] = inference.STARTGAME
        from_idx = torch.zeros(1, dtype=torch.long)
        sample = (seed_ids, from_idx)
        input_names = ["input_ids", "from_idx"]
        output_names = ["from_logits", "to_logits", "promo_logits"]
        dynamic_axes = {"input_ids": {1: "T"}}
    elif token_mode == "classic":
        wrapper = ClassicExport(model)
        seed_ids = torch.zeros(1, T, dtype=torch.long)
        sample = (seed_ids,)
        input_names = ["input_ids"]
        output_names = ["logits"]
        dynamic_axes = {"input_ids": {1: "T"}}
    else:
        print(f"ERROR: unsupported token_mode={token_mode}", file=sys.stderr)
        sys.exit(4)

    tmp_path = args.out + ".fp32.tmp.onnx"
    print(f"Tracing & exporting ONNX (token_mode={token_mode}, opset={args.opset}) -> {tmp_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            sample,
            tmp_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    import onnx

    print("Adding metadata_props ...")
    m = onnx.load(tmp_path)

    def add_meta(k, v):
        p = m.metadata_props.add()
        p.key = str(k)
        p.value = str(v)

    add_meta("token_mode", token_mode)
    add_meta("block_size", block_size)
    add_meta("vocab_size", vocab_size)
    add_meta("n_embd", n_embd)
    add_meta("n_head", n_head)
    add_meta("n_layer", n_layer)
    if n_kv_heads is not None:
        add_meta("n_kv_heads", n_kv_heads)
    add_meta("arch_version", "1")
    onnx.save(m, tmp_path)

    final_path = args.out
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print("Quantizing weights (dynamic int8) ...")
        quantize_dynamic(tmp_path, final_path, weight_type=QuantType.QInt8)
        os.remove(tmp_path)
    elif args.fp16:
        from onnxconverter_common import float16
        print("Converting weights to float16 ...")
        m_in = onnx.load(tmp_path)
        m16 = float16.convert_float_to_float16(
            m_in,
            keep_io_types=True,
            disable_shape_infer=True,
        )
        onnx.save(m16, final_path)
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, final_path)

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\nDone -> {final_path}")
    print(f"  size: {size_mb:.1f} MB")
    print(f"  token_mode={token_mode}  block_size={block_size}  vocab_size={vocab_size}")
    print(f"  n_embd={n_embd}  n_head={n_head}  n_kv_heads={n_kv_heads}  n_layer={n_layer}")
    print("\nNow open chess_full.html and pick this .onnx via the per-side file picker.")


if __name__ == "__main__":
    main()

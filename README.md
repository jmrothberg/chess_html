# Chess (Human / Search / LLM)

Full-rules chess in the browser. Choose **White Player** and **Black Player** independently: human clicks, **search** (engine), or **LLM** (local bridge).

## Play from GitHub

These links load `chess.html` straight from the repo—no clone required for **human** and **search** play.

- **GitHub Pages:** [Play on jmrothberg.github.io](https://jmrothberg.github.io/Games/chess/chess.html)
- **Raw mirror (raw.githack):** [Play via raw.githack](https://raw.githack.com/jmrothberg/Games/main/chess/chess.html)

**Why two links?** **GitHub Pages** is the normal place to host this repo: same `jmrothberg.github.io/Games/...` URLs as your other games, correct `Content-Type`, and it serves whatever your Pages build publishes (usually the default branch you configured). **raw.githack** is a fallback that proxies the **raw file from `main`** on GitHub and serves it as a real web page (opening the raw blob URL directly often downloads or mis-handles HTML). It is useful if Pages is slow to update, you want a link that always tracks `main`, or you are sharing the same pattern as older “Quick Links” in the root README. For day-to-day play, either link is fine for this single-file game.

**Search play** — Select *Search* on one or both sides. The built-in minimax + alpha-beta (iterative deepening, MVV-LVA, quiescence) runs in the page at depths 1–7. It works on the links above.

**LLM play** — The LLM option needs a local PyTorch model and the HTTP bridge; it does **not** run from a static GitHub URL alone. The LLM dropdown stays disabled until the server is running.

### Checkpoint formats (`Chess_Inference copy.py`)

`.pth` files are opened with `load_model_file`, which inspects `hyperparameters` and `model_state_dict` keys, then sets `model._token_mode` so `chess_server.py` calls the right decoder.

| Mode | What it is | How the loader tends to choose it |
|------|------------|-----------------------------------|
| **Four-token** | `ChessModel` with role heads (`head_color`, `head_from`, `head_to`, `head_promo`) and a ~**140**-token vocab (color + from + to + promo roles per ply). | `head_color` in the checkpoint **or** `format_version >= 2`, if not overridden by classic rules. |
| **Classic (1-token)** | **One** token per full move: `lm_head` over a large move vocabulary + specials (`create_classic_move_to_idx`). | `token_mode == 'classic'`, **or** `format_version >= 3` with `lm_head`, **or** basic **`TransformerModel`** (only `lm_head`; loader forces classic so the server does not use role heads). |
| **MobileLLM** | `MobileLLMModel`: RMSNorm + SwiGLU blocks, single **`lm_head`**. | Keys contain `rms_1` or `swiglu`. Decoding follows stored `token_mode` (usually classic-style). |

**Unsupported:** checkpoints with legacy **`from_head`** (factorized v1) — the loader prints an error and skips them.

The browser sends the usual **UCI** history string (`<STARTGAME> e2e4 e7e5 …`); both formats consume that text — tokenization into roles vs single move tokens is handled inside inference.

### Training repo (where these checkpoints come from)

Checkpoints are produced by the separate **Chess** training project. That repo documents data (parquet `Moves` / `Result`), per-game training, and the exact checkpoint JSON layout.

- **Overview:** [Chess_with_GPThelp_to_write — README.md](https://github.com/jmrothberg/Chess_with_GPThelp_to_write/blob/main/README.md) — classic vs 4-token modes, vocab sizes, architecture summary.
- **Technical reference:** [README_CHESS_PER_GAME.md](https://github.com/jmrothberg/Chess_with_GPThelp_to_write/blob/main/README_CHESS_PER_GAME.md) — move-id math for classic mode, role table for 4-token, `hyperparameters` / `tokenizer` fields inside `.pth`, inference pipeline description.

Trainer and browser stack use the same conceptual split:

| | **Classic (1 token / move)** | **4-token** |
|---|------------------------------|-------------|
| **Vocab** | 20,160 moves + 5 specials → **20,165** | **140** (roles + specials) |
| **Typical `format_version` in checkpoint** | **3** | **2** |
| **Typical `block_size`** | 128 tokens (128 half-moves) | 512 tokens (= 128 half-moves × 4) |

`Chess_Inference copy.py` here is a portable sibling of the trainer’s inference module; `chess_server.py` loads `.pth` files the same way and routes classic vs four-token decoding for the web UI.

### Running the LLM bridge (optional)

1. Clone this repo and put `.pth` checkpoints in `chess/Chess_LLM_models copy/` (large files; not stored in git).
2. Ensure `Chess_Inference copy.py` sits in this `chess/` folder next to `chess_server.py` (same layout the server expects).
3. Install PyTorch, then:

```bash
python3 "chess/chess_server.py"
```

4. Open [http://localhost:5858/chess.html](http://localhost:5858/chess.html) so `/api/models` and `/api/predict_move` are available to the page.

The server is stdlib `http.server` plus PyTorch; it serves `chess.html` and returns top candidate moves from the selected checkpoint.

## Standalone web play (`chess_full.html`) — no Python at runtime

`chess_full.html` is a single-file alternative to `chess.html` that runs the LLM **entirely in the browser**. No `chess_server.py`, no PyTorch, no localhost. Open it as a local file (`file://…/chess_full.html`) or host it on GitHub Pages — both work identically.

The catch: browsers can't read PyTorch `.pth` pickle files, so each checkpoint must be **converted to ONNX once** with the included `convert_pth_to_onnx.py`. After that, the user picks the `.onnx` per side via a file picker; everything else (tokenization, generation loop, top-k sampling) is JavaScript.

### One-time conversion

`convert_pth_to_onnx.py` understands both **classic** and **4-token** checkpoints (see "Checkpoint formats" above). It auto-detects the architecture, traces a single ONNX graph that exposes the right output heads, and embeds metadata (`token_mode`, `block_size`, `vocab_size`, …) into the file so the HTML needs no sidecar JSON.

```bash
# Install the export-side deps once (must match the Python that runs the script):
python3 -m pip install --user torch onnx onnxconverter-common onnxruntime

# Convert with full CLI:
python3 chess/convert_pth_to_onnx.py \
  --pth "chess/Chess_LLM_models copy/<file>.pth" \
  --out "chess/Chess_LLM_models copy/<file>.onnx"

# Or just type the command and let Finder pick the .pth (Terminal-friendly):
python3 chess/convert_pth_to_onnx.py
```

Flags:

| Flag | Effect | Typical size |
|------|--------|--------------|
| *(default)* | Trace → ONNX → fp16 weights (Cast nodes kept in fp32 for compatibility) | ~½ of fp32 |
| `--no-fp16` | Keep fp32 weights | full size; safest for parity testing |
| `--int8` | ONNX Runtime dynamic int8 quantization (overrides `--fp16`) | ~¼ of fp32 |
| `--force-export` | Re-trace even if a `.fp32.tmp.onnx` from a prior run exists | — |

The 12-layer / 512-emb checkpoint converts to roughly 80 MB (fp16). The 24-layer / 1024-emb checkpoints land at 1.5–1.8 GB (fp16) — close to ONNX's 2 GB single-file protobuf limit; use `--int8` (~400–800 MB) if you bump into that ceiling. A crashed final-conversion attempt leaves a `*.fp32.tmp.onnx` next to the target; re-running with the same `--out` reuses it and skips the slow trace+export step.

### Playing in `chess_full.html`

1. Open `chess_full.html` in Chrome or Edge (file picker + WebGPU work best there).
2. For each side you want the LLM to play, click the **Choose File** button under that side and pick a converted `.onnx`. The status line should turn green with the model's `token_mode` and context length.
3. Switch the side's player dropdown to **LLM** and click **New Game**.

Inference uses `onnxruntime-web@1.20.1` from a CDN (one-time fetch, ~3 MB). WebGPU is used automatically on Chrome/Edge; Firefox and Safari fall back to WASM (slower but works).

### `.pth` checkpoint structure (for sharing chess models in this format)

The trainer ([Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write)) saves a single Python pickle containing roughly:

```
{
  'model_state_dict': { ... per-layer weight tensors ... },
  'hyperparameters': {
    'vocab_size':    140 | 20165,         # 4-token vs classic
    'n_embd':        512 | 1024,          # model width
    'n_head':        8 | 16,              # query heads in MultiQueryAttention
    'n_kv_heads':    2 | 4,               # shared KV heads (memory savings)
    'n_layer':       12 | 24,             # transformer blocks
    'block_size':    512 (4-token) | 128 (classic),
    'dropout':       0.0,                 # ignored at inference
    'format_version':2 (4-token) | 3 (classic),
    'token_mode':    '4token' | 'classic',# explicit on newer trainers
  },
  'tokenizer': { 'token_string': id, ... }    # optional; both modes are deterministic and rebuildable
}
```

Inside `model_state_dict` the loader expects (for the **4-token** mode used by most current checkpoints):

| Key prefix | What it is |
|------------|------------|
| `token_embedding_table.weight` (V, C) | Token embeddings (V = vocab) |
| `position_embedding_table.weight` (block_size, C) | Learned position embeddings |
| `blocks.<i>.rms_1.weight`, `blocks.<i>.rms_2.weight` (C,) | Pre-attn / pre-FFN RMSNorm gains |
| `blocks.<i>.attn.q_proj.weight` (C, C) | Query projection |
| `blocks.<i>.attn.kv_proj.weight` (2 · n_kv_heads · head_dim, C) | Combined K+V projection |
| `blocks.<i>.attn.out_proj.weight` (C, C) | Attention output projection |
| `blocks.<i>.attn.q_norm.weight`, `k_norm.weight` (head_dim,) | Optional QK-Norm gains (newer checkpoints only) |
| `blocks.<i>.swiglu.{w1,w2,w3}.weight` (4C, C) / (4C, C) / (C, 4C) | SwiGLU FFN |
| `rms_final.weight` (C,) | Final RMSNorm |
| `head_color.weight` (2, C), `head_from.weight` (64, C), `head_to.weight` (64, C), `head_promo.weight` (5, C), `emb_from.weight` (64, C) | **4-token role heads** (these are what makes a checkpoint "4-token"; absent in classic checkpoints) |
| `lm_head.weight` (V, C) | **Classic** mode only — single shared head over the full move vocab |

`Chess_Inference copy.py:load_model_file` walks these keys to choose the right Python class (`ChessModel(token_mode=…)` or, for legacy formats, `MobileLLMModel` / `TransformerModel`). It tolerates the `module.` and `_orig_mod.` prefixes left by `DataParallel` and `torch.compile`.

To produce a checkpoint another instance of this app can load: train a `ChessModel` with the same architecture in the trainer repo and save:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters':  hyperparameters_dict,
    'tokenizer':        tokenizer_dict,   # optional
}, '<name>.pth')
```
The loader will figure out the rest.

## Files here

| File | Role |
|------|------|
| `chess.html` | Game UI + in-browser search engine; LLM via `chess_server.py` |
| `chess_full.html` | **Standalone** UI + in-browser ONNX inference; no server needed |
| `chess_server.py` | Local PyTorch bridge for `chess.html`'s `LLM` mode |
| `Chess_Inference copy.py` | Loads `.pth` checkpoints; generates candidate moves (classic / 4-token) |
| `convert_pth_to_onnx.py` | One-time `.pth` → `.onnx` exporter for `chess_full.html` |

---

Repos: **this game** — [jmrothberg/Games](https://github.com/jmrothberg/Games) · **training** — [jmrothberg/Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write)

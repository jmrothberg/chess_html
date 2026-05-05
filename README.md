# Chess (Human / Search / LLM)

Full-rules chess in the browser. Choose **White Player** and **Black Player** independently: human clicks, **search** (engine), or **LLM** (transformer model).

Two ways to run the LLM:

1. **`chess.html`** — talks to a local Python bridge (`chess_server.py`) that loads `.pth` checkpoints with PyTorch. Easiest for development; needs Python.
2. **`chess_full.html`** — fully standalone. Runs ONNX inference in the browser via `onnxruntime-web` + WebGPU. **No Python at runtime, no server.** Works as a `file://` page or hosted on GitHub Pages. The user picks a converted `.onnx` per side via a file picker.

---

## Play from GitHub (no setup, search engine only)

These links load `chess.html` straight from the repo. Human and search play work; LLM is disabled until a server is running.

- **GitHub Pages:** [Play on jmrothberg.github.io](https://jmrothberg.github.io/Games/chess/chess.html)
- **Raw mirror (raw.githack):** [Play via raw.githack](https://raw.githack.com/jmrothberg/Games/main/chess/chess.html)

The same applies to `chess_full.html`, with the bonus that you can also play **LLM** there once you've converted a `.pth` to `.onnx` — the file picker reads from your local disk; nothing has to be hosted.

**Search play** uses iterative-deepening alpha-beta minimax with quiescence and MVV-LVA ordering at depths 1–7.

---

## Path A — `chess.html` + Python bridge

1. Clone this repo and put `.pth` checkpoints in `chess/Chess_LLM_models copy/` (large; not stored in git).
2. Install PyTorch:
   ```bash
   python3 -m pip install --user torch
   ```
3. Start the bridge:
   ```bash
   python3 "chess/chess_server.py"
   ```
4. Open <http://localhost:5858/chess.html>. The LLM dropdown lists every `.pth` it found.

The server is stdlib `http.server` plus PyTorch; it serves the HTML at `/` and exposes `/api/models` and `/api/predict_move`.

---

## Path B — `chess_full.html`, standalone

No Python at runtime. Two artifacts: the HTML, and one `.onnx` per checkpoint you want to play.

### Step 1 — convert each `.pth` to `.onnx` (one time)

Install the conversion-side dependencies into the same Python that will run the converter:

```bash
python3 -m pip install --user torch onnx onnxconverter-common onnxruntime
```

Then run the converter. You can drive it three ways:

```bash
# 1. Zero-arg / Terminal-friendly. Pops a Finder dialog to pick one or many .pth
#    files. Outputs land next to each .pth with the same stem. A "Conversion
#    complete" dialog appears at the end.
python3 chess/convert_pth_to_onnx.py

# 2. Single file via flags.
python3 chess/convert_pth_to_onnx.py \
  --pth "chess/Chess_LLM_models copy/<file>.pth" \
  --out "chess/Chess_LLM_models copy/<file>.onnx"

# 3. Just --pth (output auto-named next to it).
python3 chess/convert_pth_to_onnx.py --pth "chess/Chess_LLM_models copy/<file>.pth"
```

Flags:

| Flag | Effect | Typical size |
|------|--------|--------------|
| *(default)* | Trace → ONNX → fp16 weights, with Cast nodes kept in fp32 for portability | ~½ of fp32 |
| `--no-fp16` | Keep fp32 weights | full size; safest for parity testing |
| `--int8` | ONNX Runtime dynamic int8 quantization (overrides `--fp16`) | ~¼ of fp32 |
| `--force-export` | Re-trace+export even if a `.fp32.tmp.onnx` from a prior run exists | — |
| `--opset N` | ONNX opset version (default 17) | — |
| `--seed-len N` | Sample sequence length used for tracing (default 16) | — |

The 12-layer / 512-emb checkpoint converts to roughly 80 MB (fp16). 24-layer / 1024-emb checkpoints land at 1.5–1.8 GB (fp16) — close to ONNX's 2 GB single-file protobuf limit. Use `--int8` (~400–800 MB) if you bump into that ceiling.

**Crash recovery:** if the final fp16/int8 step dies (e.g. missing `onnxruntime`), the metadata-tagged `.fp32.tmp.onnx` is still on disk. Re-running with the same `--out` reuses it and skips the slow trace+export step. Pass `--force-export` to override.

### Step 2 — open `chess_full.html` and play

1. Open `chess_full.html` in Chrome or Edge (WebGPU works best there; Firefox/Safari fall back to WASM).
2. For each side you want the LLM to play, click **Choose File** under that side and pick a converted `.onnx`. The status line should turn green with the model's `token_mode` and context length.
3. Switch the side's player dropdown to **LLM** and click **New Game**.

Inference uses `onnxruntime-web@1.20.1` from a CDN (one-time fetch, ~3 MB). All weights and all generation happen in the tab — no requests to your machine.

---

## Build your own chess LLM (full reference)

Anyone with PyTorch can train a checkpoint that this app will load. The shape contract is small and explicit; below is everything `Chess_Inference copy.py:load_model_file` and `convert_pth_to_onnx.py` need to recognize and convert your model.

There are **two officially supported formats** — pick one based on how you want the model to predict moves:

| | **4-token (recommended)** | **Classic (1 token / move)** |
|---|---|---|
| Tokens per ply | 4 (color, from-square, to-square, promo) | 1 (the entire move as a single token) |
| Vocab size | **140** (small, fast embeddings) | **20,165** (huge embedding table) |
| Heads in the network | 5 small role heads + a 64-row `emb_from` lookup | one big `lm_head` over the move vocab |
| `block_size` (typical) | 512 (= 128 plies × 4) | 128 plies |
| Browser inference cost / move | ~`top_k + 1` ONNX runs (one per from-candidate) | 1 ONNX run |
| Best for | Smaller param count, faster training, cleaner factorization | Larger vocab but simpler decoding |
| `format_version` to set | `2` | `3` |

The 4-token format is what most current checkpoints use. The classic format is supported for legacy compatibility.

### 1. Model architecture (`ChessModel`)

Both formats use the same trunk. The only structural difference is the head: 4-token uses five small heads + an `emb_from` lookup; classic uses one big `lm_head` (with weight tying to the input embedding). Both live in [`Chess_Inference copy.py`](Chess_Inference%20copy.py) — that's the source of truth and the file the converter imports.

```python
class ChessModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads,
                 block_size, n_layer, dropout, token_mode='4token', use_chess=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_mode = token_mode
        self.use_chess  = use_chess

        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.register_buffer('pos_indices', torch.arange(block_size))

        self.blocks = nn.ModuleList([
            Block(n_embd=n_embd, n_head=n_head, n_kv_heads=n_kv_heads, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.rms_final = RMSNorm(n_embd)

        if token_mode == 'classic':
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding_table.weight   # weight tying
        else:                                                           # 4-token
            self.head_color = nn.Linear(n_embd, 2)
            self.head_from  = nn.Linear(n_embd, 64)
            self.head_to    = nn.Linear(n_embd, 64)
            self.head_promo = nn.Linear(n_embd, 5)
            self.emb_from   = nn.Embedding(64, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(self.pos_indices[:T])
        for blk in self.blocks:
            x = blk(x, mask=None)             # mask=None → just causal
        x = self.rms_final(x)
        if self.token_mode == 'classic':
            return self.lm_head(x), None      # (B, T, vocab)
        return x, None                        # (B, T, n_embd) — heads applied externally
```

Inside each `Block`:

```python
class Block(nn.Module):           # pre-norm, RMSNorm + MultiQueryAttention + SwiGLU
    rms_1   : RMSNorm(n_embd)
    attn    : MultiQueryAttention(n_embd, n_head, n_kv_heads, dropout)
    rms_2   : RMSNorm(n_embd)
    swiglu  : SwiGLU(n_embd, hidden=4*n_embd)
    forward : x = x + attn(rms_1(x)); x = x + swiglu(rms_2(x))
```

`MultiQueryAttention`:
- `q_proj : Linear(n_embd, n_head * head_dim)` where `head_dim = n_embd // n_head`
- `kv_proj : Linear(n_embd, n_kv_heads * head_dim * 2)` — K and V together, interleaved per kv-head
- `q_norm, k_norm : RMSNorm(head_dim)` — QK-Norm (optional; older checkpoints lack these and the loader fills them with γ=1)
- `out_proj : Linear(n_embd, n_embd)`
- Causal mask is a registered buffer of `torch.tril(torch.ones(1024, 1024))`, sliced to the current `T`. The converter pre-converts this buffer to bool so ONNX-side fp16 conversion is well-typed.

`SwiGLU(in_features, hidden_features=4*in_features)`:
- `w1, w2 : Linear(in_features, hidden_features)`
- `w3 : Linear(hidden_features, in_features)`
- `forward : w3(silu(w1(x)) * w2(x))`

`RMSNorm(dim, eps=1e-5)`: `x / sqrt(mean(x**2) + eps) * weight`.

### 2. Vocabulary & tokenization

#### 4-token mode (140 tokens total)

Each ply (half-move) produces **4 tokens**, in fixed order. Constants are in [`Chess_Inference copy.py`](Chess_Inference%20copy.py) lines 642–645:

| Role | Range | Meaning |
|------|-------|---------|
| **Color** | `0..1` | `0` = WHITE, `1` = BLACK (`COLOR_OFFSET = 0`) |
| **From-square** | `2..65` | UCI from-square `a8 → h1` (`FROM_OFFSET = 2`) |
| **To-square** | `66..129` | UCI to-square (`TO_OFFSET = 66`) |
| **Promotion** | `130..134` | `none`, `q`, `r`, `b`, `n` (`PROMO_OFFSET = 130`) |
| **Specials** | `135..139` | `<STARTGAME>=135`, `<EOFG>=136`, `<PAD>=137`, `<W>=138`, `<D>=139` |

Square indexing: `(file, rank) → (8 - rank) * 8 + file`. So `a8 = 0, h1 = 63`.

Game tokenization for `"<STARTGAME> e2e4 e7e5 …"`:
```
[STARTGAME, c0, f0, t0, p0, c1, f1, t1, p1, …]
```
where `c0` is `<WHITE>` (since white moves first), `f0` is `e2 + FROM_OFFSET`, `t0` is `e4 + TO_OFFSET`, `p0` is `<PROMO:none>`. The model is trained to predict each token given everything before it.

The 140-token vocab is **deterministic** — the JS in `chess_full.html` rebuilds it from the constants above; you do not need to ship a `tokenizer` dict in your `.pth` for 4-token mode (but it's harmless if you do).

**At inference**, the browser appends one role at a time and reads the corresponding head:

| Step | Append | Read |
|------|--------|------|
| 1 | `color_tok` (we know whose turn) | `head_from(hidden[-1])` → top-K from-squares |
| 2 | `FROM + chosen_from` | `head_to(hidden[-1] + emb_from(chosen_from))` → top to-squares for this from |
| 3 (if pawn promotes) | `TO + chosen_to` | `head_promo(hidden[-1])` → argmax promo piece |

The `+ emb_from(chosen_from)` term in step 2 is the conditioning trick — the model is trained to predict the to-square given the from-square explicitly added to its hidden state.

#### Classic mode (20,165 tokens)

One token per full move: `(from_sq, to_sq, promo)` packed into a single id, plus 5 specials.

```python
def create_classic_move_to_idx():
    m = {}
    for from_sq in range(64):
        for to_sq in range(64):
            if to_sq == from_sq: continue
            to_offset = to_sq if to_sq < from_sq else (to_sq - 1)
            for promo_idx, promo_char in enumerate(['', 'q', 'r', 'b', 'n']):
                move_id  = (from_sq * 63 * 5) + (to_offset * 5) + promo_idx
                move_str = f"{file(from_sq)}{rank(from_sq)}{file(to_sq)}{rank(to_sq)}{promo_char}".upper()
                m[move_str] = move_id     # 64 * 63 * 5 = 20,160 entries
    for idx, token in enumerate(['<STARTGAME>', '<EOFG>', '<PAD>', '<W>', '<D>'], start=len(m)):
        m[token] = idx                    # ids 20160 .. 20164
    return m
```

This is also deterministic — the JS rebuilds it. Inference is one forward pass; mask the special-token ids and take top-K from `lm_head`'s last-position logits.

### 3. The `.pth` save format

The trainer writes a single Python pickle:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'vocab_size':    140 if mode == '4token' else 20165,
        'n_embd':        512,                    # model width
        'n_head':        8,                      # query heads in MultiQueryAttention
        'n_kv_heads':    2,                      # shared K/V heads (memory savings)
        'n_layer':       12,                     # transformer blocks
        'block_size':    512 if mode == '4token' else 128,
        'dropout':       0.0,                    # ignored at inference
        'format_version': 2 if mode == '4token' else 3,
        'token_mode':    mode,                   # explicit; older trainers leave this off
    },
    'tokenizer': tokenizer_dict,                 # optional — both modes are deterministic
}, 'my_model.pth')
```

`load_model_file` tolerates the `module.` and `_orig_mod.` prefixes left by `nn.DataParallel` and `torch.compile`, so you don't have to unwrap before saving.

### 4. State-dict keys (what `load_state_dict` expects)

Replace `<i>` with the block index `0 .. n_layer-1`. Shapes use `C = n_embd`, `H = head_dim = n_embd // n_head`.

| Key | Shape | Mode |
|-----|-------|------|
| `token_embedding_table.weight` | `(V, C)` | both |
| `position_embedding_table.weight` | `(block_size, C)` | both |
| `pos_indices` (buffer) | `(block_size,)` | both — auto-created by the loader if missing |
| `blocks.<i>.rms_1.weight` | `(C,)` | both |
| `blocks.<i>.rms_2.weight` | `(C,)` | both |
| `blocks.<i>.attn.q_proj.weight` | `(n_head·H, C)` = `(C, C)` | both |
| `blocks.<i>.attn.q_proj.bias` | `(C,)` | both |
| `blocks.<i>.attn.kv_proj.weight` | `(2·n_kv_heads·H, C)` | both |
| `blocks.<i>.attn.kv_proj.bias` | `(2·n_kv_heads·H,)` | both |
| `blocks.<i>.attn.q_norm.weight` | `(H,)` | optional (QK-Norm) |
| `blocks.<i>.attn.k_norm.weight` | `(H,)` | optional (QK-Norm) |
| `blocks.<i>.attn.out_proj.weight` | `(C, C)` | both |
| `blocks.<i>.attn.out_proj.bias` | `(C,)` | both |
| `blocks.<i>.attn.causal_mask` (buffer) | `(1024, 1024)` | both — auto-created |
| `blocks.<i>.swiglu.w1.weight` | `(4C, C)` | both |
| `blocks.<i>.swiglu.w1.bias` | `(4C,)` | both |
| `blocks.<i>.swiglu.w2.weight` | `(4C, C)` | both |
| `blocks.<i>.swiglu.w2.bias` | `(4C,)` | both |
| `blocks.<i>.swiglu.w3.weight` | `(C, 4C)` | both |
| `blocks.<i>.swiglu.w3.bias` | `(C,)` | both |
| `rms_final.weight` | `(C,)` | both |
| **4-token only:** `head_color.{weight,bias}` | `(2,C) / (2,)` | 4-token |
| **4-token only:** `head_from.{weight,bias}` | `(64,C) / (64,)` | 4-token |
| **4-token only:** `head_to.{weight,bias}` | `(64,C) / (64,)` | 4-token |
| **4-token only:** `head_promo.{weight,bias}` | `(5,C) / (5,)` | 4-token |
| **4-token only:** `emb_from.weight` | `(64, C)` | 4-token |
| **Classic only:** `lm_head.weight` | `(V, C)` — tied to `token_embedding_table.weight` | classic |

The loader is forgiving: missing `q_norm`/`k_norm` get default γ=1 (no-op), missing `pos_indices` gets `arange(block_size)`, missing `causal_mask` gets the standard `tril(ones(1024,1024))`.

### 5. End-to-end recipe

1. **Train** a `ChessModel` in PyTorch with the architecture above. (See the trainer at [Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write) for an example training loop, parquet data format, and per-game training script.)
2. **Save** with the dict layout in §3.
3. **Convert**:
   ```bash
   python3 chess/convert_pth_to_onnx.py     # Finder picker
   ```
   This produces `<your-name>.onnx` next to your `.pth`. The exporter:
   - Loads via `Chess_Inference copy.py:load_model_file` (which auto-detects `token_mode`).
   - Disables the optional game-mask logic (browser only ever sees one game).
   - Pre-converts `causal_mask` buffers to bool.
   - Wraps the model so a **single forward pass** returns all heads:
     - 4-token: `(input_ids, from_idx) → (from_logits, to_logits, promo_logits)`
     - classic: `(input_ids,) → logits[:, -1]`
   - Traces with `torch.onnx.export(opset=17)` and embeds metadata (`token_mode`, `block_size`, `vocab_size`, `n_embd`, `n_head`, `n_kv_heads`, `n_layer`, `arch_version`) in the ONNX `metadata_props`.
   - Optionally fp16-quantizes (`onnxconverter-common.float16` with `op_block_list=['Cast']` and `keep_io_types=True`) or int8-quantizes (`onnxruntime.quantization.quantize_dynamic`).
4. **Open** `chess_full.html`, pick the `.onnx` for a side, set that side to **LLM**, play.

### 6. Verifying conversion (parity test)

Same checkpoint via Python server vs `chess_full.html` should produce **bit-identical** top-K rankings:

```bash
# Terminal: with chess_server.py running
curl -s -X POST http://127.0.0.1:5858/api/predict_move \
  -H 'Content-Type: application/json' \
  -d '{"history":"","model":"<file>.pth","top_k":5}'
```

```js
// In chess_full.html DevTools, after loading the matching .onnx
const s = sessions.white || sessions.black;
const fn = s.tokenMode === 'classic' ? predictMoveClassic : predictMove4Token;
fn(s, '', 5).then(console.log)
```

If the move lists match, your model converted faithfully. If they diverge, either the conversion is dropping ops the model needs, or one of your custom layers isn't traceable by `torch.onnx.export`.

---

## Files here

| File | Role |
|------|------|
| `chess.html` | Game UI + in-browser search engine; LLM via `chess_server.py` |
| `chess_full.html` | **Standalone** UI + in-browser ONNX inference; no server needed |
| `chess_server.py` | Local PyTorch bridge for `chess.html`'s `LLM` mode |
| `Chess_Inference copy.py` | Loads `.pth` checkpoints; defines `ChessModel` and the generation loop |
| `convert_pth_to_onnx.py` | One-time `.pth` → `.onnx` exporter with Finder picker (run with no args) |

---

Repos: **this game** — [jmrothberg/Games](https://github.com/jmrothberg/Games) · **training** — [jmrothberg/Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write)

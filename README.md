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

## Files here

| File | Role |
|------|------|
| `chess.html` | Game UI and in-browser search engine |
| `chess_server.py` | Local bridge for LLM inference |
| `Chess_Inference copy.py` | Loads checkpoints and generates candidate moves (classic / 4-token) |

---

Repos: **this game** — [jmrothberg/Games](https://github.com/jmrothberg/Games) · **training** — [jmrothberg/Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write)

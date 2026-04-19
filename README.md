# Chess (Human / Search / LLM)

Full-rules chess in the browser. Choose **White Player** and **Black Player** independently: human clicks, **search** (engine), or **LLM** (local bridge).

## Play from GitHub

These links load `chess.html` straight from the repo—no clone required for **human** and **search** play.

- **GitHub Pages:** [Play on jmrothberg.github.io](https://jmrothberg.github.io/Games/chess/chess.html)
- **Raw mirror (raw.githack):** [Play via raw.githack](https://raw.githack.com/jmrothberg/Games/main/chess/chess.html)

**Search play** — Select *Search* on one or both sides. The built-in minimax + alpha-beta (iterative deepening, MVV-LVA, quiescence) runs in the page at depths 1–7. It works on the links above.

**LLM play** — The LLM option needs a local PyTorch model and the HTTP bridge; it does **not** run from a static GitHub URL alone. The LLM dropdown stays disabled until the server is running.

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

---

Repo: [jmrothberg/Games](https://github.com/jmrothberg/Games)

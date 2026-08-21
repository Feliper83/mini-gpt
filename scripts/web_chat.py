"""Local web UI for the mini-gpt chat session (Module 30, web variant).

Wraps the exact same ChatSession used by scripts/chat.py in a tiny local
Flask server, so you can talk to the model from a browser instead of a
terminal. Nothing here talks to the internet: Flask's dev server binds to
127.0.0.1 only, and all inference runs in-process on your CPU.

Same disclaimer as scripts/chat.py: this is a raw completion model trained
on Tiny Shakespeare, NOT an instruction-tuned assistant.

Run with:

    uv run python scripts/web_chat.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from app.inference.chat import ChatSession
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.checkpoint import load_checkpoint

CHECKPOINT_PATH = Path("checkpoints") / "tinyshakespeare.pt"
VOCAB_PATH = Path("checkpoints") / "tinyshakespeare_vocab.json"

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>mini-gpt chat</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.1rem; color: #444; }
  #log { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; height: 60vh; overflow-y: auto;
         white-space: pre-wrap; background: #fafafa; line-height: 1.4; }
  .user { color: #0a5; font-weight: 600; }
  .bot { color: #333; }
  form { display: flex; gap: 0.5rem; margin-top: 1rem; }
  input { flex: 1; padding: 0.6rem; font-size: 1rem; }
  button { padding: 0.6rem 1rem; font-size: 1rem; cursor: pointer; }
  .note { color: #888; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>mini-gpt &mdash; Tiny Shakespeare, char-level (NOT instruction-tuned)</h1>
<p class="note">This continues whatever you type in a Shakespeare-ish style.
It does not answer questions.</p>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Type text to continue...">
  <button type="submit">Send</button>
  <button type="button" id="reset">Reset</button>
</form>
<script>
const log = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");

function append(cls, text) {
  const span = document.createElement("div");
  span.className = cls;
  span.textContent = text;
  log.appendChild(span);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value;
  if (!text) return;
  input.value = "";
  append("user", "> " + text);
  const res = await fetch("/send", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text}),
  });
  const data = await res.json();
  if (data.error) {
    append("bot", "(error: " + data.error + ")");
  } else {
    append("bot", data.continuation);
  }
});

document.getElementById("reset").addEventListener("click", async () => {
  await fetch("/reset", {method: "POST"});
  log.textContent = "";
  append("bot", "(transcript cleared)");
});
</script>
</body>
</html>
"""


def create_app() -> Flask:
    """Builds the Flask app around a single shared ChatSession."""
    if not CHECKPOINT_PATH.exists() or not VOCAB_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoint or vocab. Run scripts/train_tiny_shakespeare.py first."
        )

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    loaded = load_checkpoint(CHECKPOINT_PATH)
    session = ChatSession(loaded.model, tokenizer, max_seq_len=loaded.config.max_seq_len)

    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return PAGE

    @app.post("/send")
    def send():
        text = request.get_json(force=True).get("text", "")
        try:
            continuation = session.send(text)
        except ValueError as error:
            return jsonify({"error": str(error)})
        return jsonify({"continuation": continuation})

    @app.post("/reset")
    def reset():
        session.reset()
        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    flask_app = create_app()
    print("mini-gpt web chat running at http://127.0.0.1:5000 (local only)")
    flask_app.run(host="127.0.0.1", port=5000, debug=False)

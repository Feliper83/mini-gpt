"""Module 30: Chat Interface -- an interactive REPL over a trained GPT.

IMPORTANT: this model was trained purely on next-character prediction over
raw text (Module 28: Tiny Shakespeare). It was never instruction-tuned or
fine-tuned on dialogue data (that requires RLHF, out of scope for this
project) -- so it will NOT behave like a helpful assistant. Typing a prompt
gets you a Shakespeare-style CONTINUATION of whatever you typed, not an
answer to it. That distinction is deliberate and instructive.

Run with:

    uv run python scripts/chat.py
"""

from __future__ import annotations

from pathlib import Path

from app.inference.chat import ChatSession
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.checkpoint import load_checkpoint

CHECKPOINT_PATH = Path("checkpoints") / "tinyshakespeare.pt"
VOCAB_PATH = Path("checkpoints") / "tinyshakespeare_vocab.json"

HELP_TEXT = """
Commands:
  /temp <value>   set temperature (default 0.8)
  /topp <value>   set top_p, or 'off' to disable (default 0.9)
  /topk <value>   set top_k, or 'off' to disable (default off)
  /reset          clear the conversation transcript
  /help           show this message
  /quit           exit
Anything else is appended to the transcript and continued by the model.
"""


def main() -> None:
    """Run the interactive chat REPL."""
    if not CHECKPOINT_PATH.exists() or not VOCAB_PATH.exists():
        print("Missing checkpoint or vocab. Run scripts/train_tiny_shakespeare.py first.")
        print(f"  expected: {CHECKPOINT_PATH}")
        print(f"  expected: {VOCAB_PATH}")
        return

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    loaded = load_checkpoint(CHECKPOINT_PATH)
    session = ChatSession(loaded.model, tokenizer, max_seq_len=loaded.config.max_seq_len)

    print("mini-gpt chat -- trained on Tiny Shakespeare, char-level, NOT instruction-tuned.")
    print("Type text and the model will continue it in a Shakespeare-ish style.")
    print("Type /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Bye.")
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/reset":
            session.reset()
            print("(transcript cleared)")
            continue
        if user_input.startswith("/temp "):
            session.sampling.temperature = float(user_input.removeprefix("/temp "))
            print(f"(temperature = {session.sampling.temperature})")
            continue
        if user_input.startswith("/topp "):
            value = user_input.removeprefix("/topp ")
            session.sampling.top_p = None if value == "off" else float(value)
            print(f"(top_p = {session.sampling.top_p})")
            continue
        if user_input.startswith("/topk "):
            value = user_input.removeprefix("/topk ")
            session.sampling.top_k = None if value == "off" else int(value)
            print(f"(top_k = {session.sampling.top_k})")
            continue

        try:
            continuation = session.send(user_input)
        except ValueError as error:
            print(f"(error: {error})")
            continue
        print(continuation)


if __name__ == "__main__":
    main()

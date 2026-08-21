"""Interactive chat session: a running text buffer continued by a trained GPT."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.inference.generator import Generator
from app.inference.sampling import sample_from_logits
from app.inference.temperature import apply_temperature
from app.inference.top_k import apply_top_k
from app.inference.top_p import apply_top_p
from app.tokenizer.char_tokenizer import CharTokenizer
from app.transformer.gpt_model import GPTModel


@dataclass
class SamplingConfig:
    """User-adjustable sampling parameters for a chat session."""

    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = 0.9


class ChatSession:
    """Maintains a running text transcript and generates continuations from it.

    This is NOT an instruction-tuned chat assistant. The underlying model
    (Module 28's Tiny Shakespeare run) was trained purely to predict the
    next character of raw text -- it never saw question/answer or dialogue
    data, and was never fine-tuned with RLHF (the technique that turns a
    plain completion model into something that follows instructions, well
    outside this project's scope). What this session actually does is
    simpler: it keeps appending text to a growing buffer and asks the model
    to continue it -- the same operation as Module 19's `Generator`, wrapped
    in a loop that remembers what came before and lets the sampling
    strategy (Modules 20-23) be adjusted live.
    """

    def __init__(
        self,
        model: GPTModel,
        tokenizer: CharTokenizer,
        max_seq_len: int,
        sampling: SamplingConfig | None = None,
    ) -> None:
        """Builds the session.

        Args:
            model: A trained `GPTModel`.
            tokenizer: The tokenizer the model was trained with.
            max_seq_len: The model's context length.
            sampling: Sampling parameters. Defaults to temperature=0.8, top_p=0.9.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sampling = sampling or SamplingConfig()
        self.transcript = ""

    def _sample_fn(self, logits: torch.Tensor) -> int:
        """Applies the session's current sampling config to one step's logits."""
        scaled = apply_temperature(logits, self.sampling.temperature)
        if self.sampling.top_k is not None:
            scaled = apply_top_k(scaled, self.sampling.top_k)
        if self.sampling.top_p is not None:
            scaled = apply_top_p(scaled, self.sampling.top_p)
        return sample_from_logits(scaled)

    def reset(self) -> None:
        """Clears the running transcript."""
        self.transcript = ""

    def send(self, user_text: str, max_new_tokens: int = 200) -> str:
        """Appends `user_text` to the transcript and generates a continuation.

        Args:
            user_text: Text to append before generating.
            max_new_tokens: Number of new characters to generate.

        Returns:
            Only the newly generated continuation (not the full transcript).

        Raises:
            ValueError: If `user_text` (or the accumulated transcript)
                contains a character outside the tokenizer's vocabulary --
                `CharTokenizer` (Module 2) is a partial function; unlike
                BPE (Module 3), it cannot fall back to raw bytes. The
                transcript is left unchanged if this happens, so a rejected
                message doesn't permanently poison the session.
        """
        # Encode against the candidate transcript WITHOUT mutating
        # self.transcript yet: if encode() raises, we must not have already
        # appended the bad text, or every later call would re-encode (and
        # re-fail on) this same poisoned transcript until reset().
        candidate_transcript = self.transcript + user_text
        prompt_ids = self.tokenizer.encode(candidate_transcript)
        self.transcript = candidate_transcript

        generator = Generator(
            self.model, max_seq_len=self.max_seq_len, sample_fn=self._sample_fn
        )
        generated_ids = generator.generate(prompt_ids, max_new_tokens=max_new_tokens)

        new_ids = generated_ids[len(prompt_ids) :]
        continuation = self.tokenizer.decode(new_ids)
        self.transcript += continuation
        return continuation

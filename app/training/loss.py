"""Cross-entropy loss for next-token prediction (causal language modeling)."""

from __future__ import annotations

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Next-token prediction loss: negative log-likelihood of the true token.

    For one position with logits z and true class y:
        loss = -log(softmax(z)[y]) = -z[y] + log(sum_j exp(z[j]))

    Computed with the log-sum-exp trick for numerical stability (subtracting
    max(z) before exponentiating avoids overflow, without changing the
    result, since softmax and log-sum-exp are both shift-invariant).

    Averaged over every token position in the batch: with logits of shape
    (batch, seq_len, vocab_size) and targets of shape (batch, seq_len), this
    is exactly the loss `TokenDataset` (Module 4) was built to produce
    training examples for.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the mean cross-entropy loss over all positions.

        Args:
            logits: Unnormalized scores over the vocabulary.
            targets: True next-token ids.

        Shape:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            output: scalar

        Returns:
            The mean negative log-likelihood over all `batch_size * seq_len`
            token predictions.
        """
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
        flat_targets = targets.reshape(batch_size * seq_len)

        max_logits = flat_logits.max(dim=-1, keepdim=True).values
        shifted_logits = flat_logits - max_logits
        log_sum_exp = shifted_logits.exp().sum(dim=-1).log()  # log(sum_j exp(z_j - max))

        target_shifted_logits = shifted_logits.gather(
            dim=-1, index=flat_targets.unsqueeze(-1)
        ).squeeze(-1)
        log_prob_of_target = target_shifted_logits - log_sum_exp  # log(softmax(z)[y])

        return -log_prob_of_target.mean()

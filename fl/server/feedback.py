"""
fl/server/feedback.py

Feedback Dispatcher
====================
Takes the top-down signal from Z2 → Z1 and produces
a personalized feedback vector for each client.

Information flow:
    Z2.downward_message(z2)     →  global_signal  (B, d1)
    Z1.downward_message(z1)     →  client_signals (N, B, d1)
    FeedbackDispatcher.dispatch →  per_client_feedback (N, B, d1)

Why personalization matters:
    A generic global signal tells every client the same thing —
    "here is what the global model wants." But clients have different
    architectures, different data distributions, different representational
    tendencies. A client that only sees vehicles needs different feedback
    than one that only sees animals, even if the global model's coarse
    understanding is the same.

    The dispatcher blends the global signal with the client's own z1
    (which already carries cross-client context from the Transformer)
    weighted by alpha. Low alpha = gentle nudge, high alpha = strong correction.

Phase 1 (current):
    Feedback = alpha * global_signal + (1 - alpha) * client_z1
    Simple interpolation — global signal steers, local z1 anchors.

Phase 2 (later — personalization):
    Each client gets a learned context vector maintained across rounds.
    Feedback = f(global_signal, client_context_vector)
    The server learns what each client needs over time.
    Scaffolded here but not active — set personalized=False for now.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class FeedbackDispatcher(nn.Module):
    """
    Dispatches top-down feedback to each client.

    Args:
        d1           : representation dimension
        num_clients  : total number of clients (for context vectors)
        alpha        : feedback strength (0 = no feedback, 1 = full replacement)
        personalized : if True, use per-client learned context (Phase 2)
    """

    def __init__(
        self,
        d1          : int,
        num_clients : int,
        alpha       : float = 0.15,
        personalized: bool  = False,
    ):
        super().__init__()

        self.d1           = d1
        self.alpha        = alpha
        self.personalized = personalized
        self.num_clients  = num_clients

        # Phase 2: per-client context vectors (learned embeddings)
        # Not used in Phase 1 but initialized so the scaffold is ready
        if personalized:
            self.client_contexts = nn.Embedding(num_clients, d1)
            self.context_proj    = nn.Linear(d1 * 2, d1)

    def dispatch(
        self,
        global_signal  : torch.Tensor,          # (B, d1) from Z2
        client_z1s     : torch.Tensor,          # (N, B, d1) from Z1
        client_ids     : Optional[list] = None, # for personalization
    ) -> torch.Tensor:                          # (N, B, d1)
        """
        Produce per-client feedback vectors.

        Phase 1: interpolate global signal with each client's z1.
            feedback_i = alpha * global_signal + (1 - alpha) * z1_i
            
            The global signal broadcasts across all clients.
            Each client's z1 already has cross-client context from
            the Transformer, so this isn't purely local — it's
            global-signal-steered local representation.

        Args:
            global_signal : top-down from Z2, broadcast to all  (B, d1)
            client_z1s    : per-client z1 from adapter          (N, B, d1)
            client_ids    : list of client ids (Phase 2 only)

        Returns:
            feedback      : per-client feedback vectors         (N, B, d1)
        """
        N, B, d1 = client_z1s.shape

        # Broadcast global signal across all clients
        global_broadcast = global_signal.unsqueeze(0).expand(N, -1, -1)
                                                     # (N, B, d1)

        if self.personalized and client_ids is not None:
            # Phase 2: blend global signal with learned client context
            id_tensor = torch.tensor(client_ids, device=client_z1s.device)
            contexts  = self.client_contexts(id_tensor)          # (N, d1)
            contexts  = contexts.unsqueeze(1).expand(-1, B, -1)  # (N, B, d1)

            # Concatenate global signal + context, project back to d1
            combined  = torch.cat([global_broadcast, contexts], dim=-1)
                                                                  # (N, B, 2*d1)
            personalized_signal = self.context_proj(combined)    # (N, B, d1)

            feedback = (
                self.alpha * personalized_signal
                + (1 - self.alpha) * client_z1s
            )
        else:
            # Phase 1: simple alpha-weighted interpolation
            feedback = (
                self.alpha * global_broadcast
                + (1 - self.alpha) * client_z1s
            )

        return feedback                              # (N, B, d1)

    def get_client_feedback(
        self,
        feedback  : torch.Tensor,   # (N, B, d1)
        client_idx: int,
    ) -> torch.Tensor:              # (B, d1)
        """Extract feedback vector for a single client by index."""
        return feedback[client_idx]


def build_feedback_dispatcher(cfg: dict) -> FeedbackDispatcher:
    feedback_cfg = cfg["feedback"]
    return FeedbackDispatcher(
        d1           = cfg["common"]["d1"],
        num_clients  = cfg["fl"]["num_clients"],
        alpha        = feedback_cfg["alpha"],
        personalized = feedback_cfg.get("personalized", False),
    )
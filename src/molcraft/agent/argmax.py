"""Deterministic, network-free proposal and top-k selection control."""

from molcraft.errors import ConfigurationError
from molcraft.guidance.proposals import build_proposals


class ArgmaxAgent:
    """Enumerate legal edits, then repeatedly take the highest remaining probability."""

    def propose(self, graph, evidence, policy, *, samples, round_index, context_id,
                journal_dir=None, excluded=()):
        packet = build_proposals(graph, evidence, policy, excluded=excluded)
        packet.diagnostics.update(context_id=context_id, strategy="argmax",
                                  round_index=round_index, llm_calls=0,
                                  selection="probability_top_k_without_replacement")
        return packet

    def select(self, packet, budget, *, journal_dir=None):
        if type(budget) is not int or budget < 1:
            raise ConfigurationError("Selection budget must be a positive integer")
        # noop is evaluated independently as the baseline and must not consume a slot.
        ordered = sorted((c for c in packet.candidates if c.operations),
                         key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))
        return [c.candidate_id for c in ordered[:budget]]


def create_agent(strategy):
    """Import and contact the LLM provider only for the explicitly selected strategy."""
    if strategy == "argmax":
        return ArgmaxAgent(), {"api_models_http": None, "model": None, "llm_calls": 0}
    if strategy != "optagent":
        raise ConfigurationError("Unknown optimization strategy")
    from .deepseek import DeepSeekClient, ProviderError
    from .optagent import OptAgent

    client = DeepSeekClient()
    if not client.available():
        raise ProviderError("DeepSeek models endpoint did not list deepseek-flash")
    return OptAgent(client), {"api_models_http": 200, "model": client.model}

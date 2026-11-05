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


def create_agent(strategy, *, synthesis_policy=None, denovo_policy=None):
    """Import and contact the LLM provider only for the explicitly selected strategy."""
    if strategy == "argmax":
        return ArgmaxAgent(), {"api_models_http": None, "model": None, "llm_calls": 0}
    if strategy not in {"optagent", "syncraft", "denovo"}:
        raise ConfigurationError("Unknown optimization strategy")
    from .deepseek import DeepSeekClient, ProviderError
    from .optagent import OptAgent

    client = DeepSeekClient()
    if strategy == "denovo":
        from .denovo import DeNovoAgent
        agent = DeNovoAgent(client, policy=denovo_policy)
        return agent, {"api_models_http": None, "model": client.model,
                       "agent_protocol": agent.policy.protocol,
                       "chemical_evidence": agent.evidence.resource_metadata(),
                       "synthesis_evidence": "soft_product_pattern_reference_with_optional_stock_not_route_validation"}
    if not client.available():
        raise ProviderError("DeepSeek models endpoint did not list deepseek-flash")
    if strategy == "syncraft":
        from .syncraft import SynCraftAgent, UPSTREAM_COMMIT

        agent = SynCraftAgent(client, policy=synthesis_policy)
        return agent, {"api_models_http": 200, "model": client.model,
                       "agent_protocol": agent.policy.protocol_version,
                       "syncraft_source_commit": UPSTREAM_COMMIT,
                       "synthesis_evidence": "motif_and_proxy_checked_not_retrosynthesis",
                       "selection": "catalogue_first_then_other_legal_candidates"}
    return OptAgent(client), {"api_models_http": 200, "model": client.model}

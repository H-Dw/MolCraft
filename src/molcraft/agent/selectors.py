"""External agents choose IDs from a validated packet, never arbitrary SMILES."""

from typing import Protocol

from molcraft.guidance.proposals import ProposalPacket
from molcraft.errors import ConfigurationError


class EditSelector(Protocol):
    def select(self, packet: ProposalPacket, budget: int = 1) -> list[str]: ...


class GreedySelector:
    """Highest proposal probability; not a guarantee of actual loss improvement."""

    def select(self, packet: ProposalPacket, budget: int = 1) -> list[str]:
        if budget < 1:
            raise ConfigurationError("Selection budget must be positive")
        ordered = sorted(packet.candidates,
                         key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))
        return [c.candidate_id for c in ordered[:budget]]

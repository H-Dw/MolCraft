"""Auditable group and synthesis-proxy checks, not a retrosynthesis oracle."""

from dataclasses import dataclass
from functools import lru_cache
import math

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from molcraft.errors import ChemistryError, ConfigurationError


@dataclass(frozen=True)
class SynthesisPolicy:
    protocol_version: str = "syncraft-group-v2"
    focus_atoms: int = 8
    region_atoms: int = 12
    max_sa_score: float = 6.0
    max_sa_increase: float = 2.0
    sa_weight: float = 0.25
    check_conformer: bool = True
    conformer_seed: int = 0
    conformer_iterations: int = 200
    pool_multiplier: float = 2.0
    max_drafts: int = 32
    draft_batch_size: int = 2

    def __post_init__(self):
        if self.protocol_version != "syncraft-group-v2":
            raise ConfigurationError("Unknown synthesis protocol")
        for key in ("focus_atoms", "region_atoms", "conformer_iterations", "max_drafts", "draft_batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ConfigurationError(f"synthesis.{key} must be a positive integer")
        if not math.isfinite(self.pool_multiplier) or self.pool_multiplier <= 1:
            raise ConfigurationError("synthesis.pool_multiplier must exceed one")
        if (not math.isfinite(self.max_sa_score) or not 1 <= self.max_sa_score <= 10
                or not math.isfinite(self.max_sa_increase) or self.max_sa_increase < 0
                or not math.isfinite(self.sa_weight) or self.sa_weight < 0):
            raise ConfigurationError("Invalid synthesis SA thresholds or weight")
        if type(self.check_conformer) is not bool:
            raise ConfigurationError("synthesis.check_conformer must be boolean")
        if type(self.conformer_seed) is not int or not 0 <= self.conformer_seed < 2**31:
            raise ConfigurationError("Invalid synthesis conformer seed")


# These define recognizable product motifs, not certified synthetic routes.
MOTIFS = {
    "urea": ("[N][C;X3](=[O])[N]", "urea assembly / common urea building blocks"),
    "carbamate": ("[O;X2][C;X3](=[O])[N]", "carbamate formation / protecting-group chemistry"),
    "amide": ("[C;X3](=[O])[N]", "amide coupling of suitable acid and amine building blocks"),
    "substituted_amide": ("[C;X3](=[O])[N][C,c]", "N-substituted amide building blocks / coupling"),
    "thioamide": ("[C;X3](=[S])[N]", "thioamide building blocks / functional-group conversion"),
    "ester": ("[C;X3](=[O])[O;X2][C,c]", "ester formation from suitable building blocks"),
    "carboxyl": ("[C;X3](=[O])[O;H1,-1]", "carboxylic-acid building blocks / ester hydrolysis"),
    "sulfonamide": ("[S;X4](=[O])(=[O])[N]", "sulfonamide formation with suitable amines"),
    "sulfone": ("[C,c][S;X4](=[O])(=[O])[C,c]", "sulfone building blocks / sulfide oxidation"),
    "sulfoxide": ("[C,c][S;X3](=[O])[C,c]", "sulfoxide building blocks; consider sulfur stereochemistry"),
    "nitrile": ("[C;X2]#[N;X1]", "nitrile-containing building blocks"),
    "ketone": ("[C,c][C;X3](=[O])[C,c]", "carbonyl building blocks / functional-group conversion"),
    "aldehyde": ("[C;H1](=[O])", "aldehyde building blocks / alcohol oxidation"),
    "amine": ("[N;X3,X4;!a;!$(N-[C,S,P]=[O,S,N])]", "amine building blocks / alkylation / reductive amination"),
    "ether": ("[C,c][O;X2][C,c]", "ether building blocks / ether formation"),
    "alcohol": ("[C,c][O;X2;H1]", "alcohol building blocks / carbonyl reduction"),
    "thioether": ("[C,c][S;X2][C,c]", "sulfide building blocks / C-S bond formation"),
    "alkene": ("[C;X3]=[C;X3]", "alkene building blocks; preserve specified E/Z geometry"),
    "alkyl_segment": ("[C;X4][C;X4][C;X4]", "simple alkyl building blocks"),
    "alkyl_halide": ("[C;X4][F,Cl,Br,I]", "alkyl-halide building blocks"),
    "aryl_halide": ("[c][F,Cl,Br,I]", "aryl-halide building blocks / cross-coupling precursors"),
}
RING_MOTIFS = {
    "aromatic_ring": "aromatic building blocks / appropriately substituted heteroarenes",
    "ring_3": "cyclopropane building blocks; carbocycles only in this policy",
    "ring_4": "four-membered ring building blocks; check strain and substitution",
    "ring_5": "five-membered ring building blocks / suitable cyclization",
    "ring_6": "six-membered ring building blocks / suitable cyclization",
}
_PROTECTED = (set(MOTIFS) - {"amine", "alcohol", "alkene", "alkyl_segment"}) | {"aromatic_ring"}


@lru_cache(maxsize=1)
def _queries():
    result = {name: Chem.MolFromSmarts(pattern) for name, (pattern, _) in MOTIFS.items()}
    if any(q is None for q in result.values()):
        raise ConfigurationError("Invalid synthesis motif SMARTS")
    return result


def motif_catalog():
    return [{"motif": name, "smarts": pattern, "feasibility_analogy": analogy}
            for name, (pattern, analogy) in MOTIFS.items()] + [
                {"motif": name, "feasibility_analogy": analogy}
                for name, analogy in RING_MOTIFS.items()]


def groups(graph):
    molecule = graph.to_rdkit()
    ids = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
    found = set()
    for name, query in _queries().items():
        found.update((name, tuple(sorted(ids[i] for i in match)))
                     for match in molecule.GetSubstructMatches(query))
    for ring in molecule.GetRingInfo().AtomRings():
        atom_ids = tuple(sorted(ids[i] for i in ring))
        # Every ring is an integrity unit, including rings outside the product catalog.
        found.add((f"integrity_ring_{len(ring)}", atom_ids))
        if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            found.add(("aromatic_ring", atom_ids))
        elif 3 <= len(ring) <= 6:
            allowed = {"C"} if len(ring) == 3 else {"C", "N", "O", "S"}
            if all(molecule.GetAtomWithIdx(i).GetSymbol() in allowed for i in ring):
                found.add((f"ring_{len(ring)}", atom_ids))
    return [{"id": name + ":" + ",".join(map(str, atom_ids)),
             "motif": name, "atom_ids": list(atom_ids)} for name, atom_ids in sorted(found)]


def complete_regions(graph, frozen, limit=12):
    """Expand to a fixed point; discard an oversized unit instead of truncating it."""
    molecule = graph.to_rdkit()
    ids = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
    # Overlapping alkyl triplets are catalog motifs, not indivisible groups:
    # completing them transitively would swallow an entire 15/31-carbon chain.
    units = [g for g in groups(graph) if g["motif"] != "alkyl_segment"]
    distances = Chem.GetDistanceMatrix(molecule)
    result = {}
    for i, anchor in enumerate(ids):
        if anchor in frozen:
            continue
        region = {ids[j] for j, distance in enumerate(distances[i]) if distance <= 3}
        while len(region) <= limit:
            expanded = region.union(*(set(g["atom_ids"]) for g in units
                                      if set(g["atom_ids"]) & region))
            if expanded == region:
                result[anchor] = (region, [g for g in units if set(g["atom_ids"]) <= region])
                break
            region = expanded
    return result


@lru_cache(maxsize=2048)
def sa_score(smiles):
    try:
        from rdkit.Contrib.SA_Score import sascorer
        value = float(sascorer.calculateScore(Chem.MolFromSmiles(smiles)))
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        raise ConfigurationError("SynCraft mode requires RDKit Contrib SA_Score") from exc
    if not math.isfinite(value):
        raise ConfigurationError("SA_Score returned a nonfinite value")
    return value


def alerts(graph):
    molecule = graph.to_rdkit()
    found = set()
    for atom in molecule.GetAtoms():
        symbol, atom_id = atom.GetSymbol(), atom.GetAtomMapNum()
        if symbol in {"F", "Cl", "Br", "I"} and (atom.GetDegree() != 1 or atom.GetFormalCharge()):
            found.add(("charged_or_multivalent_halogen", (atom_id,)))
        if symbol == "P" and not any(b.GetBondType() == Chem.BondType.DOUBLE and
                b.GetOtherAtom(atom).GetSymbol() == "O" for b in atom.GetBonds()):
            found.add(("nonphosphoryl_phosphorus", (atom_id,)))
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        ids = tuple(sorted((a.GetAtomMapNum(), b.GetAtomMapNum())))
        symbols = {a.GetSymbol(), b.GetSymbol()}
        if bond.GetBondType() == Chem.BondType.TRIPLE and symbols not in ({"C"}, {"C", "N"}):
            found.add(("unusual_heteroatom_triple_bond", ids))
        if bond.GetBondType() == Chem.BondType.SINGLE and a.GetSymbol() == b.GetSymbol() == "O":
            found.add(("peroxide", ids))
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) == 3 and any(molecule.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring):
            found.add(("three_membered_heterocycle", tuple(sorted(
                molecule.GetAtomWithIdx(i).GetAtomMapNum() for i in ring))))
    # These are conservative design liabilities, not universal impossibility claims.
    return found


def resonance_equivalent(source, product):
    left, right = source.to_rdkit(), product.to_rdkit()
    def topology(molecule):
        return ({a.GetAtomMapNum(): (a.GetSymbol(), a.GetTotalNumHs()) for a in molecule.GetAtoms()},
                {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                 for b in molecule.GetBonds()}, Chem.GetFormalCharge(molecule))
    if topology(left) != topology(right):
        return False
    target = Chem.MolToSmiles(right)
    flags = (Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS
             | Chem.ResonanceFlags.UNCONSTRAINED_CATIONS)
    with rdBase.BlockLogs():
        return any(Chem.MolToSmiles(form) == target for form in
                   Chem.ResonanceMolSupplier(left, flags=flags, maxStructs=256))


def check_plan(plan, source, product, touched, focus):
    keys = {"liability", "transformation", "target_motifs", "replaced_source_groups", "feasibility_basis"}
    if not isinstance(plan, dict) or set(plan) != keys:
        raise ChemistryError("A complete structured synthesis plan is required")
    for name in ("liability", "transformation", "feasibility_basis"):
        if not isinstance(plan[name], str) or not 1 <= len(plan[name].strip()) <= 600:
            raise ChemistryError(f"Synthesis {name} requires a brief explanation")
    if not touched.intersection(focus):
        raise ChemistryError("The edit must touch a high-preference focus atom")
    motifs = plan["target_motifs"]
    if not isinstance(motifs, list) or len(motifs) > 4:
        raise ChemistryError("Declare zero to four product motifs; use [] for an uncatalogued transformation")
    product_groups = groups(product)
    matched = {(g["motif"], tuple(g["atom_ids"])) for g in product_groups}
    covered, seen, warnings, matched_motifs = set(), set(), [], []
    for motif in motifs:
        if not isinstance(motif, dict) or set(motif) != {"motif", "atom_ids"}:
            raise ChemistryError("Product motifs require motif and atom_ids")
        name, ids = motif["motif"], motif["atom_ids"]
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= set(product.atom_ids)):
            raise ChemistryError("Malformed product motif or unknown atom ID")
        key = name, tuple(sorted(ids))
        if key not in matched or key in seen:
            warnings.append(f"Unmatched or duplicate motif claim: {name}:{','.join(map(str, ids))}")
            continue
        seen.add(key)
        covered.update(ids)
        matched_motifs.append({"motif": name, "atom_ids": sorted(ids)})
    if not touched <= covered:
        warnings.append("Catalogue motifs do not cover every edited atom")
    product_ids = {g["id"] for g in groups(product)}
    destroyed = [g for g in groups(source) if (g["motif"] in _PROTECTED or
                 g["motif"].startswith("integrity_ring_")) and set(g["atom_ids"]) & touched
                 and g["id"] not in product_ids]
    declared = plan["replaced_source_groups"]
    if (not isinstance(declared, list) or any(not isinstance(x, str) for x in declared)
            or len(set(declared)) != len(declared)):
        raise ChemistryError("Malformed replaced_source_groups")
    if set(declared) != {g["id"] for g in destroyed}:
        warnings.append("Replaced source group declarations differ from actual groups")
    if any(not set(g["atom_ids"]) <= covered for g in destroyed):
        warnings.append("Catalogue motifs do not cover a replaced source group as a whole")
    recognized = [g for g in product_groups if g["motif"] in set(MOTIFS) | set(RING_MOTIFS)]
    actual_covered = {i for g in recognized for i in g["atom_ids"]}
    # Rank the executed chemistry, not the LLM's ability to name a known motif.
    preferred = touched <= actual_covered and all(set(g["atom_ids"]) <= actual_covered for g in destroyed)
    return {"catalog_preferred": preferred,
            "matched_motifs": matched_motifs, "catalog_warnings": warnings,
            "recognized_product_motifs": recognized,
            "actual_replaced_source_groups": [g["id"] for g in destroyed],
            "catalogue_is_acceptance_gate": False}


def assess_synthesis(source, product, policy):
    if resonance_equivalent(source, product):
        raise ChemistryError("Resonance-only representation change is not a new molecule")
    new_alerts = alerts(product) - alerts(source)
    if new_alerts:
        raise ChemistryError("New synthesis liability: " + ", ".join(name for name, _ in sorted(new_alerts)))
    before, after = sa_score(source.smiles), sa_score(product.smiles)
    if after > policy.max_sa_score or after - before > policy.max_sa_increase:
        raise ChemistryError(f"SA proxy outside budget: {before:.3f} -> {after:.3f}")
    if policy.check_conformer:
        molecule = Chem.AddHs(product.to_rdkit())
        params = AllChem.ETKDGv3()
        params.randomSeed = policy.conformer_seed
        params.maxIterations = policy.conformer_iterations
        params.numThreads = 1
        try:
            with rdBase.BlockLogs():
                status = AllChem.EmbedMolecule(molecule, params)
        except (ValueError, RuntimeError) as exc:
            raise ChemistryError("Candidate failed bounded ETKDG conformer preflight") from exc
        if status != 0:
            raise ChemistryError("Candidate failed bounded ETKDG conformer preflight")
    return {"sa_before": before, "sa_after": after, "sa_change": after-before,
            "sa_score_lower_is_easier": True, "new_alerts": [],
            "retained_alerts": [{"name": n, "atom_ids": list(ids)} for n, ids in sorted(alerts(product))],
            "conformer_preflight": "passed" if policy.check_conformer else "not_run",
            "evidence_level": "motif_and_proxy_checked", "retrosynthesis": "not_run",
            "experimentally_synthesizable": "not_established", "protocol": policy.protocol_version}

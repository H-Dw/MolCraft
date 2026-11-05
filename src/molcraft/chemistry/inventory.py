"""Versioned product-pattern evidence with optional real stock enrichment."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from molcraft.errors import ConfigurationError

INDEX_VERSION = "motif-evidence-v2"
LEGACY_INDEX_VERSION = "inventory-motifs-v1"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def chemical_context(molecule, seeds):
    """Close rings and conjugated/multiple-bond units; include attachment neighbors.

    This is context, never a whitelist restricting editable positions.
    """
    core = set(seeds)
    units = [set(r) for r in molecule.GetRingInfo().AtomRings()]
    units += [{b.GetBeginAtomIdx(), b.GetEndAtomIdx()} for b in molecule.GetBonds()
              if b.GetIsConjugated() or b.GetBondTypeAsDouble() > 1]
    while True:
        expanded = core.union(*(u for u in units if u & core))
        if expanded == core:
            break
        core = expanded
    return core, core | {a.GetIdx() for i in core for a in molecule.GetAtomWithIdx(i).GetNeighbors()}


def _fragment_query(molecule, core):
    """Exact charge/aromatic/H/degree constraints with explicit attachment ports."""
    boundary = [b.GetIdx() for b in molecule.GetBonds()
                if (b.GetBeginAtomIdx() in core) != (b.GetEndAtomIdx() in core)]
    split = Chem.FragmentOnBonds(molecule, boundary) if boundary else Chem.Mol(molecule)
    mapping = []
    fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    for frag, original in zip(fragments, mapping):
        if not core <= set(original):
            continue
        rw = Chem.RWMol(frag)
        indices, elements = [], []
        for j, old in enumerate(original):
            if old not in core:
                # The dummy isotope identifies the original external neighbor.
                # Typed ports retain charge/aromaticity and prevent two generic
                # ports from swapping across a stereocenter and matching its enantiomer.
                neighbor = molecule.GetAtomWithIdx(frag.GetAtomWithIdx(j).GetIsotope())
                aromatic = "a" if neighbor.GetIsAromatic() else "A"
                query = Chem.AtomFromSmarts(
                    f"[#{neighbor.GetAtomicNum()};D{neighbor.GetDegree()};H{neighbor.GetTotalNumHs()};{neighbor.GetFormalCharge():+d};{aromatic}]")
            else:
                atom = molecule.GetAtomWithIdx(old)
                aromatic = "a" if atom.GetIsAromatic() else "A"
                charge = atom.GetFormalCharge()
                query = Chem.AtomFromSmarts(
                    f"[#{atom.GetAtomicNum()};D{atom.GetDegree()};H{atom.GetTotalNumHs()};{charge:+d};{aromatic}]")
                # FragmentOnBonds may reorder neighbors when inserting ports.
                query.SetChiralTag(frag.GetAtomWithIdx(j).GetChiralTag())
                indices.append(j)
                elements.append(atom.GetSymbol())
            rw.ReplaceAtom(j, query)
        return Chem.MolToSmarts(rw), indices, sorted(elements)
    return None


def inventory_motifs(molecule):
    seeds = [set(r) for r in molecule.GetRingInfo().AtomRings()]
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() not in (1, 6):
            seeds.append({atom.GetIdx(), *(a.GetIdx() for a in atom.GetNeighbors())})
    if not seeds:
        seeds = [set(range(molecule.GetNumAtoms()))]
    seen = set()
    for seed in seeds:
        core, _ = chemical_context(molecule, seed)
        if any(molecule.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for i in core):
            # A cut can make different carbon side chains indistinguishable and
            # permit an enantiomer match by port permutation. Conservatively keep
            # the complete stocked molecule when the motif asserts chirality.
            core = set(range(molecule.GetNumAtoms()))
        if len(core) < 2:
            continue
        query = _fragment_query(molecule, core)
        if query and query[0] not in seen:
            seen.add(query[0])
            yield query


def _stock_records(path):
    path = Path(path)
    if path.name.lower().endswith((".sdf", ".sdf.gz")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            for row, mol in enumerate(Chem.ForwardSDMolSupplier(handle), 1):
                identifier = next((mol.GetProp(k) for k in ("Enamine_ID", "IDNUMBER", "ID", "Catalog_ID", "_Name")
                                   if mol is not None and mol.HasProp(k) and mol.GetProp(k)), None)
                yield row, identifier, mol
    elif path.suffix.lower() in (".smi", ".smiles"):
        with path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle, 1):
                fields = line.strip().split(maxsplit=1)
                if fields:
                    yield row, fields[1] if len(fields) > 1 else None, Chem.MolFromSmiles(fields[0])
    else:
        raise ConfigurationError("Inventory must be SDF, SDF.gz or SMILES (.smi/.smiles), never case JSON")


def build_index(stock, destination, *, release, source_url, templates=None,
                template_source_url=None, expected_sha256=None, expected_raw_count=None,
                expected_template_sha256=None):
    """Build an immutable SQLite artifact and a content-addressed manifest."""
    stock = Path(stock).resolve() if stock is not None else None
    destination = Path(destination).resolve()
    if not release or not source_url:
        raise ConfigurationError("Evidence release and source URL are required")
    if stock is None and templates is None:
        raise ConfigurationError("Supply reaction templates or a real stock file; never case-derived substitutes")
    digest = file_hash(stock) if stock else None
    if expected_sha256 and digest != expected_sha256:
        raise ConfigurationError("Inventory SHA-256 does not match the supplied manifest")
    template_digest = file_hash(templates) if templates else None
    if expected_template_sha256 and template_digest != expected_template_sha256:
        raise ConfigurationError("Template SHA-256 does not match the supplied manifest")
    if templates and not template_source_url:
        raise ConfigurationError("Template provenance URL is required")
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ConfigurationError("Evidence index is immutable; use a new output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".building")
    if temporary.exists():
        raise ConfigurationError("An incomplete index already occupies the build path")
    counts = Counter(raw=0, valid=0, unique=0, invalid=0, duplicate=0,
                     raw_templates=0, templates=0, invalid_templates=0, small_templates=0, unique_templates=0)
    manifest = {"protocol": INDEX_VERSION, "provider": "Enamine" if stock else "USPTO/SimpRetro", "release": release,
                "resource_mode": "inventory_enhanced" if stock else "templates_only",
                "rdkit_version": rdBase.rdkitVersion,
                "source_url": source_url, "stock_sha256": digest,
                "original_syncraft_snapshot": False,
                "standardization": "RDKit sanitize; preserve charge, H and stereochemistry; remove atom maps; single component",
                "evidence_levels": [],
                "route_validation": "not_performed", "template_frequency_filter": "not_verified"}
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
        CREATE TABLE stocks (record_id INTEGER PRIMARY KEY, vendor_id TEXT, smiles TEXT UNIQUE);
        CREATE TABLE stock_aliases (record_id INTEGER PRIMARY KEY, vendor_id TEXT, smiles TEXT);
        CREATE INDEX stock_alias_smiles ON stock_aliases(smiles);
        CREATE TABLE motifs (id TEXT PRIMARY KEY, kind TEXT, smarts TEXT, core TEXT,
                             elements TEXT, atom_count INTEGER, support_count INTEGER, example TEXT);
        CREATE INDEX motif_size ON motifs(atom_count);
        CREATE TABLE motif_elements (element TEXT, motif_id TEXT, PRIMARY KEY(element,motif_id));
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        """)

        def add(kind, smarts, core, elements, evidence):
            identifier = kind + ":" + hashlib.sha256(smarts.encode()).hexdigest()[:24]
            db.execute("INSERT INTO motifs VALUES (?,?,?,?,?,?,1,?) ON CONFLICT(id) DO UPDATE SET support_count=support_count+1",
                       (identifier, kind, smarts, json.dumps(core), json.dumps(elements), len(core), json.dumps(evidence)))
            db.executemany("INSERT OR IGNORE INTO motif_elements VALUES (?,?)",
                           [(element, identifier) for element in set(elements)])

        with rdBase.BlockLogs():
            for row, vendor_id, molecule in (_stock_records(stock) if stock else ()):
                counts["raw"] += 1
                if molecule is None or len(Chem.GetMolFrags(molecule)) != 1 or any(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()):
                    counts["invalid"] += 1
                    continue
                for atom in molecule.GetAtoms():
                    atom.SetAtomMapNum(0)
                counts["valid"] += 1
                smiles = Chem.MolToSmiles(molecule)
                db.execute("INSERT INTO stock_aliases VALUES (?,?,?)", (row, vendor_id, smiles))
                cursor = db.execute("INSERT OR IGNORE INTO stocks VALUES (?,?,?)", (row, vendor_id, smiles))
                if not cursor.rowcount:
                    counts["duplicate"] += 1
                    continue
                counts["unique"] += 1
                molecule = Chem.MolFromSmiles(smiles)
                for smarts, core, elements in inventory_motifs(molecule):
                    add("inventory_fragment", smarts, core, elements,
                        {"stock_record": row, "vendor_id": vendor_id, "stock_smiles": smiles})
                if counts["raw"] % 1000 == 0:
                    db.commit()
            if templates:
                manifest["templates"] = {"sha256": template_digest, "source_url": template_source_url,
                                         "orientation": "retrosynthetic_product_to_reactants"}
                data = json.loads(Path(templates).read_text(encoding="utf-8"))
                if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
                    raise ConfigurationError("Templates must be a list of retrosynthetic reaction SMARTS")
                counts["raw_templates"] = len(data)
                for i, reaction in enumerate(data):
                    try:
                        rxn = rdChemReactions.ReactionFromSmarts(reaction)
                        if rxn is None or rxn.GetNumReactantTemplates() != 1:
                            raise ValueError("unsupported product template")
                        query = rxn.GetReactantTemplate(0)
                        if len(Chem.GetMolFrags(query)) != 1:
                            raise ValueError("Product pattern must be a connected local structure")
                        if query.GetNumAtoms() < 2:
                            counts["small_templates"] += 1
                            continue
                        for atom in query.GetAtoms():
                            atom.SetAtomMapNum(0)
                        add("reaction_template_support", Chem.MolToSmarts(query),
                            list(range(query.GetNumAtoms())), [a.GetSymbol() for a in query.GetAtoms()],
                            {"template_record": i, "reaction_smarts": reaction,
                             "stock_reactants_verified": False, "frequency_verified": False})
                        counts["templates"] += 1
                    except (ValueError, RuntimeError):
                        counts["invalid_templates"] += 1
        if stock and not counts["unique"]:
            raise ConfigurationError("Inventory has no usable structures")
        if stock is None and not counts["templates"]:
            raise ConfigurationError("Templates have no usable connected product patterns")
        if expected_raw_count is not None and counts["raw"] != expected_raw_count:
            raise ConfigurationError(f"Stock has {counts['raw']} records, expected {expected_raw_count}")
        counts["unique_templates"] = db.execute("SELECT count(*) FROM motifs WHERE kind='reaction_template_support'").fetchone()[0]
        manifest["evidence_levels"] = (["inventory_fragment"] if counts["unique"] else []) + (
            ["reaction_template_support"] if counts["templates"] else [])
        manifest["counts"] = dict(counts)
        manifest["motifs"] = db.execute("SELECT count(*) FROM motifs").fetchone()[0]
        db.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(manifest),))
        db.commit()
        db.close()
        temporary.replace(destination)
        manifest["index_sha256"] = file_hash(destination)
        manifest["index_path"] = destination.name
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        db.close()
        temporary.unlink(missing_ok=True)
        raise


class EvidenceIndex:
    def __init__(self, manifest_path, *, inventory_manifest=None):
        self.enhancement = None
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise ConfigurationError("Chemical evidence index is not ready; supply a template manifest (Enamine is optional; no case fallback)")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        m = self.manifest
        if (m.get("protocol"), m.get("provider")) not in {
            (INDEX_VERSION, "Enamine"), (INDEX_VERSION, "USPTO/SimpRetro"), (LEGACY_INDEX_VERSION, "Enamine")
        }:
            raise ConfigurationError("Unsupported chemical evidence provenance")
        db_path = (path.parent / m["index_path"]).resolve()
        if file_hash(db_path) != m["index_sha256"]:
            raise ConfigurationError("Evidence artifact hash mismatch")
        self.db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        embedded = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        if any(m.get(k) != v for k, v in embedded.items()):
            self.db.close()
            raise ConfigurationError("Evidence manifest disagrees with embedded provenance")
        if inventory_manifest:
            try:
                self.enhancement = InventoryIndex(inventory_manifest)
            except Exception:
                self.db.close()
                raise

    @classmethod
    def from_policy(cls, policy):
        # inventory_manifest alone remains an explicit legacy stock-index choice.
        return cls(policy.evidence_manifest or policy.inventory_manifest or "",
                   inventory_manifest=policy.inventory_manifest if policy.evidence_manifest else None)

    def resource_metadata(self):
        return {"primary": self.manifest,
                "inventory_enhancement": self.enhancement.manifest if self.enhancement else None,
                "stock_loaded": bool(self.manifest["counts"]["unique"] or self.enhancement)}

    def close(self):
        if self.enhancement:
            self.enhancement.close()
        self.db.close()

    @staticmethod
    def _card(row):
        if row is None:
            raise ConfigurationError("Unknown chemical evidence ID")
        return {**dict(row), "core": json.loads(row["core"]), "elements": json.loads(row["elements"]),
                "example": json.loads(row["example"]), "route_validated": False}

    def get(self, identifier):
        row = self.db.execute("SELECT * FROM motifs WHERE id=?", (identifier,)).fetchone()
        if row is None and self.enhancement:
            return self.enhancement.get(identifier)
        return self._card(row)

    def retrieve(self, graph, tables, *, limit=6, anchor=None):
        # Retrieval is a bounded heuristic, never a chemical validity test. Match
        # elemental demand to distinct positions (including unchanged context), rather
        # than counting the best atom several times as if it could form a whole motif.
        from molcraft.agent.context import top_half_directions

        moves = tables["atom"]
        molecule = graph.to_rdkit()
        indices = {a.GetAtomMapNum(): a.GetIdx() for a in molecule.GetAtoms()}
        anchor = anchor if anchor is not None else graph.atom_ids[0]
        _, nearby = chemical_context(molecule, {indices[anchor]})
        nearby_ids = {molecule.GetAtomWithIdx(i).GetAtomMapNum() for i in nearby}
        options = {i: {element: 0.0} for i, element, _ in graph.atoms}
        for row in top_half_directions(graph, tables, set(graph.atom_ids))["atom"]:
            options[row["ids"][0]].update({label: score for label, score, _ in row["directions"]})
        preferred = sorted(options[anchor], key=lambda e: (options[anchor][e], e))
        choices = {}
        for kind in ("inventory_fragment", "reaction_template_support"):
            # Element-indexed lookup prevents the commonest carbon-only fragments
            # from consuming the entire retrieval budget for a heteroatom direction.
            placeholders = ",".join("?" for _ in preferred)
            rows = {}
            indexes = [self] + ([self.enhancement] if self.enhancement and kind == "inventory_fragment" else [])
            for index in indexes:
                query = index.db.execute(
                    f"SELECT * FROM motifs WHERE atom_count<=? AND kind=? AND id IN "
                    f"(SELECT motif_id FROM motif_elements WHERE element IN ({placeholders})) "
                    "ORDER BY support_count DESC,id LIMIT 512", (len(graph.atom_ids), kind, *preferred))
                for row in query:
                    rows.setdefault(row["id"], row)
            ranked = []
            for row in rows.values():
                card = self._card(row)
                if any(e not in moves.labels and e != "*" for e in card["elements"]):
                    continue
                used, missing, cost = set(), 0, 0.0
                elements = sorted(card["elements"], key=lambda e: (sum(e in v for v in options.values()), e))
                for element in elements:
                    available = [(i not in nearby_ids, options[i].get(element, 0), i)
                                 for i in options if i not in used and (element == "*" or element in options[i])]
                    if not available:
                        missing += 1
                        continue
                    _, score, position = min(available)
                    used.add(position)
                    cost += score
                card["retrieval"] = {"anchor": anchor, "unassigned_elements": missing,
                                     "outside_local_context": len(used - nearby_ids),
                                     "method": "bounded_distinct_position_element_assignment_not_graph_proof"}
                ranked.append(((missing, len(used - nearby_ids), cost / max(1, len(elements)),
                                -card["support_count"], card["id"]), card))
            choices[kind] = [c for _, c in sorted(ranked)]
        inventory, templates = choices["inventory_fragment"], choices["reaction_template_support"]
        # Reserve a minority of prompt cards for connection evidence if available.
        n_templates = min(len(templates), limit // 3)
        selected = inventory[:limit - n_templates] + templates[:n_templates]
        return (selected + templates[n_templates:] + inventory[limit - n_templates:])[:limit]

    def assess(self, source, product, operations, evidence_ids):
        mol = product.to_rdkit()
        source_mol = source.to_rdkit()
        touched = {i for op in operations for i in op.atom_ids}
        hits, covered, template_supported = [], set(), False
        for identifier in sorted(set(evidence_ids)):
            card = self.get(identifier)
            query = Chem.MolFromSmarts(card["smarts"])
            prior_matches = {tuple(source_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in match)
                             for match in source_mol.GetSubstructMatches(query, useChirality=True)}
            for match in mol.GetSubstructMatches(query, useChirality=True):
                core = {mol.GetAtomWithIdx(match[j]).GetAtomMapNum() for j in card["core"]}
                if not core & touched:
                    continue
                # Templates are connection evidence, not verified complete stock motifs.
                if card["kind"] == "inventory_fragment":
                    covered |= core
                mapped_match = tuple(mol.GetAtomWithIdx(i).GetAtomMapNum() for i in match)
                supports_bundle = (card["kind"] == "reaction_template_support"
                                   and mapped_match not in prior_matches
                                   and touched <= core
                                   and _template_covers_operations(query, mapped_match, operations))
                template_supported |= supports_bundle
                hits.append({"evidence_id": identifier, "evidence_type": card["kind"],
                             "atom_ids": sorted(core), "source": card["example"],
                             "supports_complete_update": supports_bundle if card["kind"] == "reaction_template_support" else touched <= core,
                             "route_validated": False})
        return {"catalog_preferred": bool(touched) and touched <= covered,
                "template_preferred": bool(touched) and template_supported,
                "matches": hits, "route_validation": "not_performed"}


def _template_covers_operations(query, mapped_match, operations):
    """A permissive/wildcard match cannot justify a specific chemical update."""
    position = {identifier: i for i, identifier in enumerate(mapped_match)}
    for op in operations:
        if op.kind == "bond":
            bond = query.GetBondBetweenAtoms(*(position[i] for i in op.atom_ids))
            if bond is None or str(bond.GetBondType()) != op.value:
                return False
        else:
            atom = query.GetAtomWithIdx(position[op.atom_ids[0]])
            if op.kind == "atom" and atom.GetSymbol() != op.value:
                return False
            if op.kind == "charge" and ("AtomFormalCharge" not in atom.DescribeQuery()
                                         or atom.GetFormalCharge() != op.value):
                return False
    return True


class InventoryIndex(EvidenceIndex):
    """Compatibility entry point for explicitly requested real Enamine stock."""

    def __init__(self, manifest_path):
        super().__init__(manifest_path)
        if self.manifest.get("provider") != "Enamine":
            self.close()
            raise ConfigurationError("Inventory enhancement must come from real Enamine stock")

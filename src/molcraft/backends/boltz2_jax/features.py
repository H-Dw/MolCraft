"""Offline, graph-preserving native Boltz feature preparation.

The native affinity SMILES standardizer and recursive dataset fallback are not
used. Native parsing/featurization is called directly with checked local assets.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import random
import shutil
import numpy as np
import yaml

from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.mapping import AtomLayout
from molcraft.errors import ConfigurationError, MappingError, MissingAssetError
from molcraft.types import PreparedComplex


def verify_assets(paths, *, require_weights=True):
    if paths is None:
        raise MissingAssetError("Explicit local model_paths are required; no assets are downloaded")
    names = ("structure_checkpoint", "affinity_checkpoint", "ccd_path") if require_weights else ("ccd_path",)
    for name in names:
        path = Path(getattr(paths, name)).expanduser()
        if not path.is_file() or path.stat().st_size == 0:
            raise MissingAssetError(f"Missing/nonempty local resource required: {name}: {path}")
    if not Path(paths.molecule_dir).expanduser().is_dir():
        raise MissingAssetError("A local native Boltz molecule directory is required")


def graph_matches_native(graph, native_mol):
    """Compare chemistry and mapped identity, ignoring serialization H policy."""
    native = MolecularGraph.from_rdkit(native_mol)
    if (native.atoms != graph.atoms or native.bonds != graph.bonds
            or native.mapped_smiles != graph.mapped_smiles):
        raise MappingError("Native preprocessing changed mapped ligand chemistry or stereochemistry")


def layout_from_native(graph, features, tokens, structure, molecules, chain_ids, spec):
    """Build raw-atom indices using native token slices and mapped molecule names."""
    from rdkit import Chem

    asym = np.asarray(features["asym_id"])[0]
    mol_type = np.asarray(features["mol_type"])[0]
    atom_to_token = np.asarray(features["atom_to_token"])[0]
    atom_mask = np.asarray(features["atom_pad_mask"])[0] > 0.5
    if len(tokens) > len(asym) or not np.array_equal(asym[:len(tokens)], tokens["asym_id"]):
        raise MappingError("Native token order disagrees with feature token order")
    ligand_chain = chain_ids[spec.ligand_chain_id]
    atom_map, token_map = {}, {}
    ligand_names = set()
    for t, token in enumerate(tokens):
        if int(token["asym_id"]) != ligand_chain:
            continue
        if mol_type[t] != 3 or int(token["atom_num"]) != 1:
            raise MappingError("Design ligand must use one native heavy atom per NONPOLYMER token")
        residue_name = str(token["res_name"])
        ligand_names.add(residue_name)
        native_mol = molecules[residue_name]
        name_to_id = {a.GetProp("name"): a.GetAtomMapNum() for a in native_mol.GetAtoms()}
        name = str(structure.atoms[int(token["atom_idx"])]["name"])
        if name not in name_to_id or name_to_id[name] <= 0:
            raise MappingError("Native ligand atom lost its stable map ID")
        stable_id = int(name_to_id[name])
        indices = np.flatnonzero((atom_to_token[:, t] > 0.5) & atom_mask)
        if len(indices) != 1 or stable_id in atom_map:
            raise MappingError("Native ligand mapping is ambiguous")
        atom_map[stable_id], token_map[stable_id] = int(indices[0]), t
    if len(ligand_names) != 1 or set(atom_map) != set(graph.atom_ids):
        raise MappingError("Native ligand identity/count does not match the requested molecule")
    graph_matches_native(graph, molecules[next(iter(ligand_names))])
    protein_tokens = (mol_type == 0) & (np.asarray(features["token_pad_mask"])[0] > 0.5)
    protein_atoms = np.flatnonzero(atom_mask & (atom_to_token[:, protein_tokens].sum(axis=1) > 0.5))
    pocket_tokens, groups = [], []
    for ref in spec.pocket:
        chain_tokens = [i for i, token in enumerate(tokens) if int(token["asym_id"]) == chain_ids[ref.chain_id]]
        chain_tokens.sort(key=lambda i: int(tokens[i]["res_idx"]))
        protein_spec = next(p for p in spec.proteins if p.chain_id == ref.chain_id)
        if len(chain_tokens) != len(protein_spec.sequence):
            raise MappingError("Protein sequence positions cannot be mapped one-to-one to native tokens")
        t = chain_tokens[ref.position - 1]
        pocket_tokens.append(t)
        groups.append(tuple(int(i) for i in np.flatnonzero((atom_to_token[:, t] > 0.5) & atom_mask)))
    element_indices = np.asarray(features["ref_element"])[0].argmax(axis=-1)
    radii = tuple(Chem.GetPeriodicTable().GetRvdw(int(element_indices[i]) + 1) for i in protein_atoms)
    return AtomLayout(graph.atom_ids, tuple(atom_map[i] for i in graph.atom_ids),
                      tuple(token_map[i] for i in graph.atom_ids), tuple(int(i) for i in protein_atoms),
                      tuple(pocket_tokens), tuple(groups), len(atom_mask), len(asym), radii)


class NativeWriter:
    def __init__(self, record, processed_dir, root, atom_mask):
        self.record, self.processed_dir, self.root = record, processed_dir, root
        self.atom_mask = np.asarray(atom_mask)

    def __call__(self, coordinates, path):
        import torch
        import boltz.main as native

        output = self.root / "native_export"
        writer = native.BoltzWriter(data_dir=self.processed_dir, output_dir=output,
                                    output_format="mmcif", boltz2=True)
        coords = np.asarray(coordinates).reshape(1, 1, -1, 3)
        pred = {"exception": False, "coords": torch.tensor(coords),
                "masks": torch.tensor(self.atom_mask).reshape(1, 1, -1),
                "confidence_score": torch.ones(1)}
        writer.write_on_batch_end(None, None, pred, None, {"record": [self.record]}, None, None)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / self.record.id / f"{self.record.id}_model_0.cif", destination)


def _reference_conformer(mol, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    heavy = Chem.Mol(mol)
    expanded = Chem.AddHs(heavy)
    expanded.RemoveAllConformers()
    options = AllChem.ETKDGv3()
    options.randomSeed = int(seed)
    if AllChem.EmbedMolecule(expanded, options) < 0:
        options.useRandomCoords = True
        if AllChem.EmbedMolecule(expanded, options) < 0:
            raise ConfigurationError("Cannot generate the ligand reference conformer")
    if AllChem.UFFHasAllMoleculeParams(expanded):
        AllChem.UFFOptimizeMolecule(expanded, maxIters=1000)
    result = Chem.RemoveHs(expanded, sanitize=False)
    conformer = result.GetConformer()
    conformer.SetProp("name", "Computed")
    conformer.SetProp("coord_generation", "ETKDGv3")
    return result


def prepare_complex(spec):
    # Native chemical features require CCD/molecules, independently of weights.
    # The model loader verifies both checkpoints before loading either network.
    verify_assets(spec.model_paths, require_weights=False)
    graph = MolecularGraph.from_smiles(spec.ligand_smiles)
    if len(graph.atom_ids) > 128:
        raise ConfigurationError("Pinned Boltz affinity supports at most 128 ligand heavy atoms")
    for protein in spec.proteins:
        if protein.msa_path and not Path(protein.msa_path).is_file():
            raise MissingAssetError(f"Local MSA file is missing: {protein.msa_path}")
        if protein.template_path and not Path(protein.template_path).is_file():
            raise MissingAssetError(f"Local template file is missing: {protein.template_path}")
        if protein.template_chain_id and not protein.template_path:
            raise ConfigurationError("template_chain_id requires a local template_path")
    try:
        import boltz.main as native
        from boltz.data import const
        from boltz.data.feature.featurizerv2 import Boltz2Featurizer, process_msa_features
        from boltz.data.module.inferencev2 import load_input
        from boltz.data.mol import load_canonicals, load_molecules
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    except ImportError as exc:
        raise ConfigurationError("Install the pinned .[boltz2] extra in a dedicated Linux environment") from exc
    temporary = TemporaryDirectory(prefix="molcraft-")
    root = Path(temporary.name)
    try:
        sequence_specs = [{"protein": {"id": p.chain_id, "sequence": p.sequence,
                                        "msa": str(Path(p.msa_path).resolve()) if p.msa_path else "empty"}}
                          for p in spec.proteins]
        sequence_specs.append({"ligand": {"id": spec.ligand_chain_id, "smiles": graph.mapped_smiles}})
        # No affinity property: that native branch standardizes the molecule.
        input_path = root / "complex.yaml"
        schema = {"version": 1, "sequences": sequence_specs}
        templates = [{"cif": str(Path(p.template_path).resolve()), "chain_id": [p.chain_id],
                      "template_id": [p.template_chain_id or p.chain_id]}
                     for p in spec.proteins if p.template_path]
        if templates:
            schema["templates"] = templates
        input_path.write_text(yaml.safe_dump(schema),
                              encoding="utf-8", newline="\n")
        # Restore host RNG state even if native preprocessing changes it.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(spec.conformer_seed)
            np.random.seed(spec.conformer_seed)
            manifest = native.process_inputs(data=native.check_inputs(input_path), out_dir=root,
                ccd_path=Path(spec.model_paths.ccd_path), mol_dir=Path(spec.model_paths.molecule_dir),
                use_msa_server=False, msa_server_url="", msa_pairing_strategy="greedy", boltz2=True)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        processed = root / "processed"
        if manifest is None:
            manifest = native.Manifest.load(processed / "manifest.json")
        if len(manifest.records) != 1:
            raise MappingError("Exactly one native complex record is required")
        record = manifest.records[0]
        input_data = load_input(record=record, target_dir=processed / "structures", msa_dir=processed / "msa",
            constraints_dir=processed / "constraints" if (processed / "constraints").exists() else None,
            template_dir=processed / "templates" if (processed / "templates").exists() else None,
            extra_mols_dir=processed / "mols", affinity=False)
        tokenized = Boltz2Tokenizer().tokenize(input_data)
        molecules = dict(load_canonicals(Path(spec.model_paths.molecule_dir)))
        molecules.update(input_data.extra_mols)
        names = set(tokenized.tokens["res_name"].tolist()) - set(molecules)
        molecules.update(load_molecules(Path(spec.model_paths.molecule_dir), names))
        chain_ids = {c.chain_name: c.chain_id for c in record.chains}
        ligand_names = {str(t["res_name"]) for t in tokenized.tokens
                        if int(t["asym_id"]) == chain_ids[spec.ligand_chain_id]}
        for name in ligand_names:
            graph_matches_native(graph, molecules[name])
            molecules[name] = _reference_conformer(molecules[name], spec.conformer_seed)
        raw = Boltz2Featurizer().process(tokenized, molecules=molecules,
            random=np.random.default_rng(spec.conformer_seed), training=False,
            max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
            compute_constraint_features=True, compute_affinity=False)
        # Native affinity uses the query-only profile, not the structure MSA average.
        raw.update(process_msa_features(tokenized, random=np.random.default_rng(spec.conformer_seed),
            max_seqs_batch=1, max_seqs=1, pad_to_max_seqs=False, affinity=True))
        features = {k: np.asarray(v)[None] for k, v in raw.items()}
        features["affinity_token_mask"] = ((features["asym_id"] == chain_ids[spec.ligand_chain_id])
                                             & (features["token_pad_mask"] > 0.5)).astype(np.float32)
        layout = layout_from_native(graph, features, tokenized.tokens, tokenized.structure,
                                     molecules, chain_ids, spec)
        from .injection import NativeVocabulary, verify_chemical_features
        vocabulary = NativeVocabulary.from_boltz(const)
        vocabulary.validate()
        verify_chemical_features(graph, features, layout, vocabulary)
        features["msa"] = np.eye(const.num_tokens, dtype=np.float32)[features["msa"].astype(int)]
        writer = NativeWriter(record, processed, root, features["atom_pad_mask"])
        return PreparedComplex(spec, graph, features, layout, writer,
                               {"vocabulary": vocabulary, "record": record}, temporary)
    except Exception:
        temporary.cleanup()
        raise

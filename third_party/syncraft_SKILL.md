---
name: syncraft-design
description: |
  Edit unsynthesizable, computationally-generated drug candidate molecules into
  synthesizable analogs that retain binding to a target protein pocket. Invoke
  when the user has SMILES from a 3D-aware generative model (Pocket2Mol, ResGen,
  TargetDiff, DiffSBDD, LIGAN, AR, or any de novo source) and asks for
  "synthesizable rescue", "make X synthesizable", "fix synthesis cliff", "edit
  to synthesizable", "lower SA score", or similar. The skill docks the inputs
  at the target pocket with AutoDock Vina, calls a reasoning LLM
  (Gemini-2.5-Pro, Gemini-3.1-Pro, or DeepSeek-V4-Pro) to predict atom-level
  edit sequences that preserve the PLIP interaction profile while removing
  synthesis liabilities, re-docks the edited products, and returns a ranked
  list of (original, edited, scores, similarity, reasoning) tuples. Use ONLY
  when the inputs are already-generated molecules and a defined receptor pocket
  exist. Do NOT use for: de novo generation, full retrosynthesis route
  planning, free-energy calculations, or covalent docking.
---

# SynCraft Skill — molecular synthesizability editing

## When to invoke

Trigger this skill when the user input matches **all three**:
1. A list of SMILES from a 3D-aware generative model, or a path to a `.smi` / `.txt` file containing one SMILES per line.
2. A target protein: a 4-letter PDB ID, a `.pdb` file path, or a known pocket descriptor.
3. An intent expressed as "make synthesizable", "edit to easier synthesis", "fix synth cliff", "lower SA", "drug-like rescue", "drug-like edit", or equivalent.

**Counter-indications** (do NOT invoke):
- User asks to *generate* new molecules from a pocket → use a generative-model skill instead.
- User asks for full retrosynthesis route → use a dedicated retrosynthesis-planning tool instead.
- User asks for binding-mode analysis only (no editing) → use a PLIP/docking-only skill.

## What it does

```
INPUT                                OUTPUT
─────────────────────────            ─────────────────────────────────
list[SMILES] (generated)       ─→    list[EditResult]
PDB id or .pdb file           ─→     ├── original_smiles
[pocket residues OR box]       ─→    ├── original_dock_score (Vina kcal/mol)
[choice of rescue_llm]         ─→    ├── edited_smiles
                                     ├── edited_dock_score
                                     ├── similarity_to_original (Tanimoto)
                                     ├── edit_operations (list[op, atom_idx])
                                     ├── edit_reasoning (LLM explanation)
                                     └── success (bool)
```

## Single-call entry point

```python
from syncraft import edit_for_synthesizability

results = edit_for_synthesizability(
    molecules=[                                   # list of SMILES
        "NC(=O)[C@@H]1CCC(C(=O)N[C@H]2OC(=O)[C@@H](O)[C@@H](O)[C@H]2NNC2C=CC=CC=C2Cl)C1",
        # ...
    ],
    protein_pdb="7L12",                           # PDB ID or local .pdb
    pocket_residues=None,                         # optional explicit residue list
    rescue_llm="deepseek-v4-pro",                 # gemini-2.5-pro | gemini-3.1-pro | deepseek-v4-pro
    pass_k=5,                                     # rescue attempts per molecule
    output_dir="./edited_hits",
    seed=42,
)
```

Returns `list[EditResult]`, one per input molecule. Results also written as JSON to `output_dir/results.json` plus per-pose PDBs in `output_dir/poses/`.

## Pipeline (4 stages, no retrosynthesis)

| Stage | Tool | Time per molecule |
|---|---|---|
| 1. Dock original molecule at pocket | AutoDock Vina | ~5-15 s |
| 2. Extract PLIP interactions from docked pose | PLIP | ~1 s |
| 3. LLM rescue: edit sequence preserving PLIP profile | reasoning LLM | ~5-15 s per pass |
| 4. Apply edits, validate, re-dock edited products | RDKit + Vina | ~10-20 s |

For 100 candidates × pass_k=5 with Vina-cpu=2 × 8 workers: ~30-60 min wall.

## Supported LLMs

| Model | License | Local-able | Note |
|---|---|---|---|
| `gemini-2.5-pro` | API only | no | paper headline; thinking_mode=high |
| `gemini-3.1-pro` | API only | no | newer, better at long reasoning chains |
| `deepseek-v4-pro` | **MIT (open weights)** | yes | reproducible long-term; HF: `deepseek-ai/DeepSeek-V4-Pro` |

Default: `deepseek-v4-pro` (long-term reproducibility per NMI editor's reproducibility requirement).

## Required environment

- Conda env with: rdkit ≥ 2022, vina 1.2.x, meeko 0.5+, plip ≥ 2.3, biopython, numpy, pydantic
- API keys (one of): `GEMINI_API_KEY` (for gemini-*), local vLLM/SGLang endpoint (for deepseek-v4-pro)
- See `setup.sh` for one-line bootstrap

## Required resources at runtime

- A receptor `.pdbqt` (auto-prepared from `.pdb` if not provided)
- A Vina box config: 6 floats (center_x/y/z + size_x/y/z); auto-derived from `pocket_residues` if given
- An "analysis PDB" (receptor with HETATM trimmed) for PLIP

## Outputs

```
output_dir/
├── results.json          # list[EditResult] serialized
├── results.csv           # tabular view (one row per molecule)
├── summary.md            # human-readable summary with top-10 ranked hits
├── poses/
│   ├── orig_NN__rep0.pdb
│   └── edit_NN_pXX__rep0.pdb
└── plip/
    └── *.json            # per-pose interaction profile
```

## Examples

See `examples/`:
- `examples/01_mpro_targetdiff.ipynb` — 47 TargetDiff candidates → SARS-CoV-2 Mpro (7L12) → 5-step demo
- `examples/02_kinase_p2m.ipynb` — Pocket2Mol candidates → RIPK1 (7YDX) kinase rescue
- `examples/quickstart.py` — minimal 20-line invocation

## Citation

If you use this skill in published work, cite:

```bibtex
@article{li2026syncraft,
  title={SynCraft: Guiding Large Language Models to Predict Edit Sequences for Molecular Synthesizability Optimization},
  author={Li, Junren and ... and Lai, Luhua},
  journal={Nature Machine Intelligence},
  year={2026},
  note={under revision}
}
```

## Skill version

`v0.1.0` (2026-05-17). Initial release coupled to NMI paper revision.

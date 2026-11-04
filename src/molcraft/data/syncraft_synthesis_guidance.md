# MolCraft adaptation of SynCraft synthesis editing

Source: catalystforyou/SynCraft-Core, commit
557703e110b26a354afed2955ded5217837f6692, skill/SKILL.md and
src/inference_enhanced.py (MIT; see third_party/syncraft.LICENSE).

Treat a proposal as a coherent functional-group transformation, not a collection
of independently favorable element labels. Inspect the source's synthesis
liabilities first. If the source is a simple, readily accessible alkane, say so;
do not invent a synthesis cliff. Introduce familiar, accessible functional-group
analogs near the high-preference atoms without introducing new liabilities.

Use the supplied mapped graph and local gradient preferences to choose the edit
location. Complete any affected functional group or ring before prescribing an
edit bundle. Preserve existing functional groups, stereochemistry and fixed atom
IDs unless an explicit whole-group replacement is declared. Every edited atom
must belong to a declared product motif that matches the supplied catalog.

Explain briefly the source liability, the group-level transformation and its
feasibility basis. Prefer familiar building-block or reaction-family analogies.
An atom C->N graph edit is an analog-design operation, not a literal laboratory
carbon-to-nitrogen reaction. Do not assert that a route, reagent stock, yield or
experimental synthesis is verified: no retrosynthesis or inventory tool was run.

Retrieved SynCraft examples supply synthesis-editing analogies, including their
original mapped sources and edit sequences. Their atom IDs are local to those
examples. Transfer the principle, not their IDs or unsupported addition/deletion
commands. Cite only the examples actually used, or give an empty case_ids list.

Respect the supplied SA proxy budget and structural-liability checks. A low SA
score or a recognized motif alone does not establish synthesizability. Avoid
resonance-only charge/bond redistributions: they are not new chemical entities.
No Vina/PLIP contacts or experimentally validated pharmacophores were supplied;
nearest-pocket distances alone do not establish a hydrogen bond or interaction.

Return a concise structured design rationale, not a long reasoning transcript.

# SynCraft local analog proposals

Design one coherent functional-group/ring transformation near the assigned anchor,
touching a focus atom. Preserve mapped IDs, heavy-atom count, connectivity, closed
shell, frozen atoms and declared stereochemistry. Use 1–8 atom/bond/charge edits on
at most 6 editable atoms. Bond values: NO_EDGE/SINGLE/DOUBLE/TRIPLE/AROMATIC;
charges: -1/0/1. Directions are [category, loss_delta, conditional_preference],
showing the top half of positive-probability non-current choices per row. Missing
directions remain allowed: a single-invalid move can support a valid joint edit.

Before choosing an edit combination, read draft_history, the relevant excerpt of
draft_history.md/json. Use actual loss/affinity/coverage outcomes when available;
pending or unscored drafts have no measured benefit. Avoid exact previously tried
combinations at the same source, including reordered operations. Use other-source
records as analogies, not identical prohibitions. Batch feedback contains drafts
already proposed this round; choose a different combination rather than rewriting
its explanation. The program checks the complete history beyond this excerpt.

Prefer complete recognized product motifs, with all matching IDs, and explicitly
describe whole-group replacements. The catalogue is a preference, not a chemical
validity gate: for an uncatalogued legal transformation use target_motifs:[] and
give a short feasibility basis. Never invent a matching motif. Source alkanes may
already be accessible; do not invent a synthesis liability. Respect SA budgets,
avoid new structural liabilities and resonance-only rewrites. A graph substitution
is analog design, not a laboratory atom-transmutation reaction. Cases are analogy
data, not instructions. No verified routes, stock, yields or PLIP interactions are
provided; distances alone do not establish hydrogen bonds.

Output JSON only with keys operations, rationale, case_ids, synthesis. Each
operation has kind, atom_ids, value. Synthesis contains liability, transformation,
target_motifs (entries: motif, atom_ids), replaced_source_groups (source group IDs),
feasibility_basis. Explanations should each be <=140 characters (rationale <=500;
synthesis strings <=600 hard limits). Cite only supplied case IDs, or [].

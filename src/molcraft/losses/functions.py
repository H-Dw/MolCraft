"""Shared JAX losses on raw atom coordinates and explicit token masks."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from rdkit import Chem

from molcraft.chemistry.vocabulary import ATOMIC_NUMBERS

VDW_RADII = np.array([Chem.GetPeriodicTable().GetRvdw(z) for z in ATOMIC_NUMBERS],
                     dtype=np.float32)


def contact_loss(logits, bins, cutoff=8.0):
    """[pocket residues, ligand atoms, distance bins], DBMol Eq. 3-6."""
    probabilities = jax.nn.softmax(logits, axis=-1)
    contact = jnp.sum(probabilities * (jnp.asarray(bins) < cutoff), axis=-1)
    pairs = -jnp.log(contact + 1e-8)
    return 0.5 * jnp.mean(jnp.min(pairs, axis=1)) + 0.5 * jnp.min(pairs)


def aggregate_affinity(heads):
    """Match probability averaging, not logit averaging, for an ensemble."""
    values = jnp.stack([jnp.asarray(h["affinity_pred_value"]).reshape(()) for h in heads])
    logits = jnp.stack([jnp.asarray(h["affinity_logits_binary"]).reshape(()) for h in heads])
    p = jnp.mean(jax.nn.sigmoid(logits))
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    ensemble_logit = jnp.log(p) - jnp.log1p(-p)
    # Do not unnecessarily saturate the exact single-head logit.
    if len(heads) == 1:
        ensemble_logit = logits[0]
    return {"affinity_pred_value": values.mean(), "affinity_logits_binary": ensemble_logit,
            "affinity_probability_binary": p, "head_values": values, "head_logits": logits}


def confidence_loss(confidence):
    return (-confidence["ligand_iptm"] - 0.3 * confidence["complex_iplddt"]
            + 0.3 * confidence["complex_ipde"] + 0.1 * confidence["complex_pde"]
            - 0.3 * confidence["confidence_score"] - 0.1 * confidence["ptm"])


def geometry_loss(coordinates, atom_weights, layout, objective):
    xyz = jnp.asarray(coordinates).reshape((-1, layout.total_atoms, 3))[0]
    ligand = xyz[jnp.asarray(layout.ligand_atom_indices)]
    temperature = objective.softmin_temperature
    terms = []
    for group in layout.pocket_atom_groups:
        delta = xyz[jnp.asarray(group)][:, None, :] - ligand[None, :, :]
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-8)
        nearest = -temperature * logsumexp(-distance.reshape(-1) / temperature)
        terms.append(jax.nn.softplus(nearest - objective.geometry_cutoff))
    pocket_term = jnp.mean(jnp.stack(terms))
    protein = xyz[jnp.asarray(layout.protein_atom_indices)]
    delta = ligand[:, None, :] - protein[None, :, :]
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-8)
    ligand_radii = atom_weights @ jnp.asarray(VDW_RADII)
    radii = ligand_radii[:, None] + jnp.asarray(layout.protein_vdw_radii)[None, :]
    overlap = jax.nn.relu(objective.clash_scale * radii - distance)
    clash_term = jnp.mean(overlap * overlap)
    return pocket_term + clash_term, {"geometry_pocket": pocket_term, "geometry_clash": clash_term}


def evaluate_objective(output, inputs, layout, objective):
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    metrics = {}
    affinity = contact = confidence = geometry = zero
    if objective.affinity_weight:
        affinity = output.affinity["affinity_pred_value"] - output.affinity["affinity_logits_binary"]
    if objective.contact_weight:
        logits = output.distogram_logits[jnp.asarray(layout.pocket_token_indices)]
        logits = logits[:, jnp.asarray(layout.ligand_token_indices)]
        contact = contact_loss(logits, output.distogram_bins, objective.contact_cutoff)
    if objective.confidence_weight:
        confidence = confidence_loss(output.confidence)
    # Zero optimization weight must not turn measured geometry into a fake zero.
    geometry, extra = geometry_loss(output.coordinates, inputs.atoms, layout, objective)
    metrics.update(extra)
    total = (objective.affinity_weight * affinity + objective.contact_weight * contact
             + objective.confidence_weight * confidence + objective.geometry_weight * geometry)
    metrics.update({"loss": total, "affinity_loss": affinity, "contact_loss": contact,
                    "confidence_loss": confidence, "geometry_loss": geometry})
    metrics.update({"weighted_affinity": objective.affinity_weight * affinity,
                    "weighted_contact": objective.contact_weight * contact,
                    "weighted_confidence": objective.confidence_weight * confidence,
                    "weighted_geometry": objective.geometry_weight * geometry})
    return total, metrics

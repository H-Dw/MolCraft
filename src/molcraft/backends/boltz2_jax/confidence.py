"""JAX adaptation of Boltz frame masks and pTM; see third_party notices.

Nearest-neighbour frame choice and collinearity masks are deliberately discrete,
as in upstream. They are not a soft coordinate gradient estimator.
"""

import jax
import jax.numpy as jnp


def frame_mask(coordinates, features, layout):
    xyz = jnp.asarray(coordinates).reshape((-1, layout.total_atoms, 3))[0]
    frames = jnp.asarray(features["frames_idx"]).reshape((-1, layout.total_tokens, 3))[0]
    if len(layout.atom_ids) >= 3:
        atom_indices = jnp.array(layout.ligand_atom_indices)
        ligand = xyz[atom_indices]
        squared = jnp.sum((ligand[:, None] - ligand[None, :]) ** 2, axis=-1)
        nearest = jnp.argsort(squared, axis=-1, stable=True)[:, :3]
        ligand_frames = atom_indices[nearest[:, jnp.array([1, 0, 2])]]
        frames = frames.at[jnp.array(layout.ligand_token_indices)].set(ligand_frames)
    frame_xyz = xyz[frames]
    v1, v2 = frame_xyz[:, 1] - frame_xyz[:, 0], frame_xyz[:, 1] - frame_xyz[:, 2]
    norm1 = jnp.sqrt(jnp.sum(v1 * v1, axis=-1, keepdims=True) + 1e-20)
    norm2 = jnp.sqrt(jnp.sum(v2 * v2, axis=-1, keepdims=True) + 1e-20)
    angle = jnp.abs(jnp.sum((v1 / (norm1 + 1e-6)) * (v2 / (norm2 + 1e-6)), axis=-1))
    return ((angle < 0.9063) & (norm1[:, 0] > 1e-2) & (norm2[:, 0] > 1e-2))


def tm_metrics(pae_logits, coordinates, features, layout):
    pad = jnp.asarray(features["token_pad_mask"])[0]
    valid_frame = frame_mask(coordinates, features, layout) * pad
    asym = jnp.asarray(features["asym_id"])[0]
    pair = valid_frame[:, None] * pad[:, None] * pad[None, :]
    inter = pair * (asym[:, None] != asym[None, :])
    ligand = jnp.zeros_like(pad).at[jnp.array(layout.ligand_token_indices)].set(1)
    mol_type = jnp.asarray(features["mol_type"])[0]
    protein = (mol_type == 0) * pad
    ligand_pair = inter * (ligand[:, None] * protein[None, :] + protein[:, None] * ligand[None, :])
    bins = (jnp.arange(pae_logits.shape[-1]) + 0.5) * (32.0 / pae_logits.shape[-1])
    d0 = 1.24 * (jnp.maximum(pad.sum(), 19) - 15) ** (1 / 3) - 1.8
    expectation = jnp.sum(jax.nn.softmax(pae_logits, axis=-1) / (1 + (bins / d0) ** 2), axis=-1)

    def score(mask):
        return jnp.max(jnp.sum(expectation * mask, axis=-1) / (mask.sum(axis=-1) + 1e-5))

    return {"ptm": score(pair), "iptm": score(inter), "ligand_iptm": score(ligand_pair),
            "valid_frame_fraction": jnp.mean(valid_frame)}

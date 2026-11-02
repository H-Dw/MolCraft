"""Ligand-safe Mosaic/Joltz forward; no protein backbone views are constructed.

Adapted from Mosaic 5937f371 and Joltz ed0f0425 under their MIT licenses.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from molcraft.losses.functions import aggregate_affinity
from molcraft.sampling.core import integrate
from molcraft.sampling.boltz2 import boltz2_denoiser
from molcraft.types import ModelOutput
from .injection import bond_embedding
from .confidence import tm_metrics


class InitialEmbedding(NamedTuple):
    s_inputs: object
    s_init: object
    z_init: object
    relative_position_encoding: object


def embed_inputs(model, features):
    s_inputs = model.input_embedder(features)
    s_init = model.s_init(s_inputs)
    z = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    relative = model.rel_pos(features)
    z = z + relative + model.token_bonds(features["token_bonds"])
    if model.bond_type_feature:
        z = z + bond_embedding(features, model.token_bonds_type.weight)
    z = z + model.contact_conditioning(features)
    return InitialEmbedding(s_inputs, s_init, z, relative)


def run_trunk(model, features, initial, passes, key):
    import joltz

    state = joltz.TrunkState(s=jnp.zeros_like(initial.s_init), z=jnp.zeros_like(initial.z_init))

    def body(carry, _):
        state, key = carry
        state = jax.tree.map(jax.lax.stop_gradient, state)
        state, key = model.trunk_iteration(state, initial, features, key=key, deterministic=True)
        return (state, key), None

    (state, _), _ = jax.lax.scan(body, (state, key), None, length=passes)
    return state


def confidence_forward(module, initial, state, xyz, features, logits, key):
    """Retain upstream hard geometry bins; replace every bond lookup path."""
    s_inputs = module.s_inputs_norm(initial.s_inputs)
    s = state.s if module.no_update_s else module.s_norm(state.s)
    if module.add_s_input_to_s:
        s = s + module.s_input_to_s(s_inputs)
    z = module.z_norm(state.z)
    if module.add_z_input_to_z:
        z = z + module.rel_pos(features) + module.token_bonds(features["token_bonds"])
        if module.bond_type_feature:
            z = z + bond_embedding(features, module.token_bonds_type.weight)
        z = z + module.contact_conditioning(features)
    z = z + module.s_to_z(s_inputs)[:, :, None] + module.s_to_z_transpose(s_inputs)[:, None, :]
    if module.add_s_to_z_prod:
        z = z + module.s_to_z_prod_out(module.s_to_z_prod_in1(s_inputs)[:, :, None]
                                      * module.s_to_z_prod_in2(s_inputs)[:, None, :])
    rep = features["token_to_rep_atom"] @ xyz
    delta = rep[:, :, None] - rep[:, None, :]
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-8)
    hard_bins = jnp.sum(distance[..., None] > module.boundaries, axis=-1).astype(jnp.int32)
    z = z + module.dist_bin_pairwise_embed(hard_bins)
    mask = features["token_pad_mask"]
    s, z = module.pairformer_stack(s, z, mask=mask, pair_mask=mask[:, :, None] * mask[:, None, :],
                                   key=key, deterministic=True)
    return module.confidence_heads(s=s, z=z, x_pred=xyz, d=distance, feats=features,
                                   pred_distogram_logits=logits[None])


def forward(bundle, features, layout, sampler, noise, mode, passes=1, key=None):
    key = jax.random.key(0) if key is None else key
    model = bundle.structure
    initial = embed_inputs(model, features)
    state = run_trunk(model, features, initial, passes, key)
    logits = model.distogram_module(state.z)[0, :, :, 0, :]
    denoise, mask = boltz2_denoiser(model, features, initial, state)
    xyz = integrate(denoise, mask, sampler, noise)[None]
    if mode == "trunk_only":
        xyz = jax.lax.stop_gradient(xyz)
    confidence = confidence_forward(model.confidence_module, initial, state, xyz, features,
                                    logits, jax.random.fold_in(key, 5))
    metrics = {name: getattr(confidence, name)[0] for name in
               ("complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")}
    metrics.update(tm_metrics(confidence.pae_logits[0], xyz, features, layout))
    interface = jnp.where(jnp.isclose(metrics["ligand_iptm"], 0), metrics["ptm"], metrics["ligand_iptm"])
    metrics["confidence_score"] = (4 * metrics["complex_plddt"] + interface) / 5
    affinity_inputs = bundle.affinity_embedder(features, affinity=True)
    ligand = features["affinity_token_mask"].astype(jnp.float32)
    protein = (features["mol_type"] == 0) * features["token_pad_mask"]
    cross = (ligand[:, :, None] * protein[:, None, :] + protein[:, :, None] * ligand[:, None, :]
             + ligand[:, :, None] * ligand[:, None, :])
    heads = [head(s_inputs=affinity_inputs, z=state.z * cross[..., None], x_pred=xyz,
                  feats=features, key=key, deterministic=True) for head in bundle.affinity_heads]
    affinity = aggregate_affinity(heads)
    bins = jnp.linspace(2.0, 22.0, logits.shape[-1])
    return ModelOutput(logits, bins, xyz, metrics, affinity)

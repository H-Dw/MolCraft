"""Machine-readable source audit; not a claim of calibration in MolCraft."""

from dataclasses import asdict

DBMOL = "https://arxiv.org/html/2607.19237v1#A3.SS1"
MOSAIC = "https://github.com/escalante-bio/mosaic/tree/5937f37164bcb28bc6f1000f10134145438e5971"
BOLTZ = "https://github.com/escalante-bio/boltz/tree/7ea718a1440ecbe8e2418317192b501544147edd"


def weight_audit(config):
    adopted = {"affinity_weight": 1.0, "contact_weight": 2.5, "confidence_weight": 2.5,
               "contact_cutoff": 8.0}
    objective = asdict(config.objective)
    rows = {}
    for name, value in objective.items():
        if name == "protocol":
            continue
        direct = name in adopted and value == adopted[name]
        rows[name] = {"value": value, "origin": "paper_direct" if direct else "local_engineering_setting",
                      "source": DBMOL if direct else None, "calibrated_in_molcraft": False}
    return {"objective_protocol": config.objective.protocol, "coefficients": rows,
            "contact_components": {"pocket_mean": 0.5, "anchor": 0.5, "source": DBMOL},
            "affinity_components": {"prediction": 1.0, "binding_logit": -1.0, "source": DBMOL},
            "confidence_components": {"ligand_iptm": -1., "complex_iplddt": -.3,
                "complex_ipde": .3, "complex_pde": .1, "confidence_score": -.3, "ptm": -.1,
                "source": DBMOL,
                "masks": {"ligand_iptm": "local ligand-protein cross-chain adaptation",
                          "complex_iplddt": "native ligand/interface/noninterface weights 20/10/1 at 8A",
                          "complex_ipde": "native probability-weighted all cross-chain pairs",
                          "complex_pde": "native probability-weighted nonself pairs"},
                "units": {"ligand_iptm": "0..1", "complex_iplddt": "0..1", "complex_ipde": "angstrom",
                          "complex_pde": "angstrom", "confidence_score": "0..1", "ptm": "0..1"},
                "upstream_aggregation_source": BOLTZ},
            "confidence_score": {"plddt": .8, "interface": .2, "source": BOLTZ},
            "sampling": {"origin": "upstream_implementation", "source": MOSAIC,
                "design_steps": config.sampling_steps, "design_passes": config.trunk_passes,
                "evaluation_steps": config.evaluation_sampling_steps,
                "evaluation_passes": config.evaluation_trunk_passes,
                "sampler_parameters": "loaded Boltz diffusion configuration"},
            "proposal": {"origin": "local_engineering_setting", "temperature": config.policy.temperature,
                "category_priors": {k: getattr(config.policy, k + "_prior") for k in ("atom", "bond", "charge")},
                "probabilities_calibrated": False},
            "synthesis": {"sa_weight_active": config.synthesis.sa_weight if config.optimization.strategy == "syncraft" else 0.0,
                          "origin": "local_engineering_setting", "route_validated": False}}

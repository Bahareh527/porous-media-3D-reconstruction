"""Generate maintained notebook templates without archiving older versions.

This tool never accesses the author\'s original research folders.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SETUP = """from pathlib import Path
import numpy as np

REPO_ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'pyproject.toml').is_file() and (p / 'configs' / 'demo.json').is_file()), None)
if REPO_ROOT is None:
    raise RuntimeError('Start Jupyter inside the repository after installing the package.')
"""


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(title, content):
    return {
        "cells": [
            cell(
                "markdown",
                f"# {title}\n\nMaintained workflow. Default examples are synthetic smoke tests—not thesis results. Install `pip install -e '.[learning,notebook]'` first. Only maintained notebook versions are included in this repository.",
            ),
            cell("code", SETUP),
            *content,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    templates = {}
    templates["01_data_preparation.ipynb"] = notebook(
        "01 — Validated 3D data preparation",
        [
            cell(
                "markdown",
                "A dry run is the default. All class IDs remain integer. Real-data filenames, spatial splits, optional gaps and voxel size are edited in a JSON configuration—not in utility code. Scaling is fitted only on the training region.",
            ),
            cell(
                "code",
                "from porous_media.config import load_config\nfrom porous_media.preparation import prepare\n\nCONFIG_PATH = REPO_ROOT / 'configs' / 'demo.json'\nSAVE_PATCHES = False\nconfig = load_config(CONFIG_PATH)\nmetadata = prepare(config, REPO_ROOT, write=SAVE_PATCHES)\nprint('Synthetic:', metadata['synthetic_only'])\nprint('Regions:', metadata['regions'])\nprint('Scaler:', metadata['normalization'])\nfor split in ('train', 'validation', 'test'):\n    print(split, sum(p['split'] == split for p in metadata['patches']))",
            ),
            cell(
                "markdown",
                "For real Bentheimer data select `configs/bentheimer.json`, verify label semantics and set the physical voxel size from provider metadata. The recorded 128/128/544 split is supported, but adjoining spatial regions can remain correlated. A nonzero `gap_slices` is available for a new leakage-sensitive experiment. Enabling patch writing creates paired `.npz` files and a checksummed manifest; existing runs are never silently overwritten.",
            ),
        ],
    )
    modes = [
        ("02_vox2vox_reconstruction.ipynb", "02 — Binary and multimineral Vox2Vox", "vox2vox"),
        (
            "03_unet_uresnet_segmentation.ipynb",
            "03 — U-Net and residual U-Net segmentation",
            "segmentation",
        ),
        (
            "04_hybrid_unet_gan_reconstruction.ipynb",
            "04 — Segmentation-informed hybrid GAN",
            "hybrid",
        ),
    ]
    for filename, title, mode in modes:
        templates[filename] = notebook(
            title,
            [
                cell(
                    "markdown",
                    "Run one actual small forward/backward pass on 16³ synthetic patches. This checks the maintained model and objective, not convergence or publication metrics. For real training, use the manifest produced by Notebook 01.",
                ),
                cell(
                    "code",
                    f"from porous_media.demos import run_learning_demo\n\nresult = run_learning_demo('{mode}')\nresult",
                ),
                cell(
                    "markdown",
                    f"### Real experiment\n\nUse the explicit CLI: `porous-media train --manifest outputs/bentheimer-128/manifest.json --mode {mode} --output outputs/{mode}-run --epochs 100 --filters 64 --depth 4`. Review the configuration before starting. For segmentation, `--residual` selects the native reference residual U-Net. The native reference model is not checkpoint-compatible with the original `segmentation_models_3D` ResNet18 implementation. Hybrid training additionally requires `--segmentation-checkpoint`, `--encoder-layer`, and a decoder filter count matching the chosen encoder's downsampling depth; no checkpoint or layer is silently assumed.",
                ),
                cell(
                    "markdown",
                    "Vox2Vox uses conditional paired labels/images and a coherent least-squares or BCE adversarial objective plus optional L1 reconstruction. Categorical IDs remain separate from normalized model conditioning. The hybrid uses an explicitly selected frozen segmentation encoder and an unconditional image discriminator; its default L1 weight is zero, matching an adversarial-only reference objective. Alternative historical losses are not silently redefined in later cells.",
                ),
            ],
        )
    templates["05_image_and_pore_network_metrics.ipynb"] = notebook(
        "05 — Measured image and pore statistics",
        [
            cell(
                "code",
                "from porous_media.preparation import synthetic_rock\nfrom porous_media.volumes import IntensityScaler\nfrom porous_media.metrics import reconstruction_metrics, pore_mask, porosity, axial_two_point\n\nDEMO = True\nif DEMO:\n    raw, labels = synthetic_rock((16, 16, 16))\n    reference = IntensityScaler.fit(raw).transform(raw)\n    prediction = np.clip(reference + np.random.default_rng(42).normal(0, 0.02, reference.shape), -1, 1)\nelse:\n    reference = np.load(REPO_ROOT / 'outputs' / 'reference.npy', allow_pickle=False)\n    prediction = np.load(REPO_ROOT / 'outputs' / 'prediction.npy', allow_pickle=False)\n\nreport = reconstruction_metrics(reference, prediction, data_range=2.0)\nthreshold = 0.0  # Explicit shared policy for this synthetic example, not a rock pore threshold.\nreference_pores = pore_mask(reference, threshold=threshold)\npredicted_pores = pore_mask(prediction, threshold=threshold)\nprint('Synthetic:', DEMO)\nprint(report)\nprint('Porosity:', porosity(reference_pores), porosity(predicted_pores))\nlag, probability = axial_two_point(reference_pores, axis=0, max_lag=8)\nprint('Lag:', lag, 'Two-point probability:', probability)",
            ),
            cell(
                "markdown",
                "Use a physically justified pore-label or threshold policy for real images and record it. Fixed data range must match the intensity convention (2 for [-1,1]). Axial two-point probability is not a radial correlation estimate. SNOW2 extraction and OpenPNM flow require the optional flow environment, provider voxel size in meters, validated topology and explicit geometry/phase models; see `docs/flow-validation.md`. Do not substitute typed published numbers for computed measurements.",
            ),
        ],
    )
    templates["06_relative_permeability_analysis.ipynb"] = notebook(
        "06 — Unit-safe flow and saturation analysis",
        [
            cell(
                "code",
                "from porous_media.flow import darcy_permeability, invasion_limits, occupancy_and_saturation\n\n# Mathematical smoke example: SI values, not a thesis network.\npermeability = darcy_permeability(flow_rate_m3_s=1e-12, length_m=1e-3, viscosity_pa_s=1e-3, area_m2=1e-6, pressure_drop_pa=1.0)\nprint('Synthetic Darcy example:', permeability)\npore_sequence = np.array([0, 1, 2, -1])\nthroat_sequence = np.array([1, 2, -1])\npore_volume = np.ones(4)\nthroat_volume = np.ones(3)\ncurve = []\nfor limit in invasion_limits(pore_sequence, throat_sequence, points=20):\n    _, _, saturation = occupancy_and_saturation(pore_sequence, throat_sequence, pore_volume, throat_volume, int(limit))\n    curve.append((int(limit), saturation))\nprint('Sequence limits and saturation:', curve)",
            ),
            cell(
                "markdown",
                "### Real relative permeability\n\n`porous_media.flow.relative_permeability(network, nonwetting_phase, wetting_phase, invasion, inlets, outlets, points=20)` operates on caller-validated OpenPNM inputs. All extracted geometry must already use meters; viscosity is Pa·s and pressure is Pa. See `docs/flow-validation.md` before simulating. The numerical smoke example does not validate SNOW2 extraction, multiphase boundary conditions or thesis permeability values. Uninvaded negative sequences are explicitly excluded, limits cannot have a zero range step, and single-phase reference flow is checked before division.",
            ),
        ],
    )
    for name, value in templates.items():
        previous = NOTEBOOKS / name
        for index, notebook_cell in enumerate(value["cells"]):
            notebook_cell["id"] = f"cell-{index:03d}"
        previous.write_text(
            json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    path = NOTEBOOKS / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["curated_sha256"] = hashlib.sha256(
            (NOTEBOOKS / row["notebook"]).read_bytes()
        ).hexdigest()
        row["source"] = "Maintained package workflow"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Generated six maintained notebooks.")


if __name__ == "__main__":
    main()

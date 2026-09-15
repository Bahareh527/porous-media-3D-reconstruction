# Notebook guide

Six maintained guides use shared code in `src/porous_media/`. All run independently in fresh kernels on small synthetic inputs by default; none depends on another notebook's hidden variables.

| Notebook | Maintained scope |
| --- | --- |
| [01](01_data_preparation.ipynb) | Validated configuration, training-only scaling, aligned patches; dry-run by default |
| [02](02_vox2vox_reconstruction.ipynb) | Actual conditional Vox2Vox training step |
| [03](03_unet_uresnet_segmentation.ipynb) | Actual native U-Net and residual U-Net steps |
| [04](04_hybrid_unet_gan_reconstruction.ipynb) | Actual hybrid step with an explicitly selected frozen encoder |
| [05](05_image_and_pore_network_metrics.ipynb) | Measured image metrics, porosity and axial two-point probability |
| [06](06_relative_permeability_analysis.ipynb) | SI Darcy arithmetic and uninvaded-sequence handling |

Install `pip install -e ".[learning,notebook]"`, start Jupyter inside the repository and select that environment's kernel. Default examples are **smoke checks, not thesis results**. Follow [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for real-data instructions. Optional network extraction and multiphase flow need a validated physical setup.

Only maintained notebooks are included in the current repository. Original research folders were not edited. [manifest.csv](manifest.csv) records notebook checksums. Native segmentation and hybrid architectures remain explicitly documented as reference implementations, not verified exact reproductions of pretrained-backbone or published models.

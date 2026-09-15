# Code review and maintained implementation

## Scope and scientific traceability

Only the maintained examples backed by `src/porous_media/` are included in the current repository. Older experimental notebooks were removed at the author's request. Original research folders on the author's device remain unchanged. The thesis document and abstract are not included in the repository; publication references and rights statements remain.

This is a functional software baseline, not a new claim of reproducing published experiments. Historical checkpoints and full source datasets are absent. Reported results in the README remain attributed to the thesis/publications.

## Findings and corrections

| Historical problem | Maintained approach |
| --- | --- |
| Later cells redefine generators, losses and training steps; one Vox2Vox loss has incompatible call signatures | Named model factories and one GAN trainer with explicit objective selection |
| Fixed training/validation iteration counts, assumed class weights and patch shapes | Finite paired manifest iterators, sample-weighted epoch losses and explicit model geometry |
| Independently normalized splits; masks treated like grayscale intensities | Training-only fitted intensity scaler; categorical IDs separate from model conditioning |
| Labels silently rewritten and patch-grid resizing changes voxel geometry | Explicit noncascading class mapping; slicing without spatial resampling |
| Missing variables, manually selected layer offsets and assumed hybrid feature dimensions | Explicit encoder layer and shape-inferred decoder depth |
| Mixed sigmoid/BCE/least-squares/WGAN alternatives | Linear discriminator with coherent BCE-with-logits or least-squares loss; no implicit WGAN claim |
| Manually typed metrics and undefined experiment arrays | Metrics computed from supplied arrays, fixed data range, explicit pore-mask policy |
| Negative invasion sequences counted as invaded; possible zero range step | Finite nonnegative occupancy and unique sequence limits |
| Inconsistent length/area/viscosity conversions; randomized extracted geometry | Explicit SI Darcy inputs and preserved caller-supplied network geometry |
| Notebook syntax checks only | Regression tests, actual forward/backward steps, checkpoint round-trip and independent notebook execution |

## Architecture and objective decisions

Vox2Vox retains the historical downsampling/upsampling and concatenating bottleneck pattern. Instance normalization is a serializable native Keras layer, replacing the legacy TensorFlow Addons dependency. Activations are explicit (LeakyReLU slope 0.2). This does not guarantee compatibility with historical weights or exact numerical equivalence.

Segmentation offers native U-Net and additive residual U-Net with sparse categorical cross-entropy. The original research experiments also used `segmentation_models_3D`, ResNet18 and focal/Dice losses. The native factories are not those pretrained-backbone models, and must not be labelled exact journal reproductions without a separate architecture reconciliation.

The hybrid takes an explicit named segmentation feature layer, freezes its encoder, infers its spatial downsampling and reconstructs a single-channel tanh volume. It uses an unconditional image discriminator, with adversarial-only loss by default (`l1_weight=0`). Vox2Vox uses paired label conditioning and L1 weight 100 by default. These values are explicit configurable reference settings, not inferred publication hyperparameters.

The training API stores optimizer state, epoch, model checkpoints and measured history. `GanTrainer.restore` supports recovery when supplied the matching model architecture and objective. The CLI intentionally refuses nonempty run folders; it does not silently resume or overwrite.

## Data and measurement conventions

Axis order is the loaded `(z, y, x)` order. NetCDF blocks concatenate along axis 0; registration/orientation must be verified externally. Train/validation/test regions do not overlap. Adjacent micro-CT slices may still be correlated; optional spatial gaps support a new leakage-sensitive experiment, rather than guaranteeing independence.

Scaling fits minimum/maximum on the training region and clips other regions to [-1,1]; both parameters and the clipping policy are recorded. Patches preserve the source voxel grid. Training patch extraction drops an incomplete trailing edge, while inference explicitly covers edges and overlap-averages predictions. Prepared patches pair images and labels in one file with origins and SHA-256 checksums; loading is batchwise, but loading/scaling source volumes still needs whole-volume RAM.

PSNR and SSIM require a declared fixed data range (2 for [-1,1]). Perfect reconstruction has infinite PSNR, represented as JSON `null` with `perfect_reconstruction=true`. Segmentation macro scores exclude classes absent from both arrays. Two-point probability is axial, not radial. Real pore thresholds must be scientifically justified and recorded; the synthetic threshold is not a calibrated rock threshold.

## Verification limits

The suite checks numerical invariants, dry-run/no-overwrite behavior, paired batches, checksums, overlap inference, model updates, frozen encoders, serialization and optimizer recovery. Notebooks are executed in separate fresh kernels without committing outputs. Optional NetCDF I/O is validated when its dependency is installed; optional OpenPNM multiphase integration is separately described in the flow guide.

These checks do not establish convergence, dataset registration, pore-network extraction quality, physically correct multiphase boundaries, or agreement with thesis permeability/segmentation/reconstruction results. Before reporting a new experiment, preserve input hashes, environment export, commit, seeds, splits, voxel size, architecture/loss, checkpoint and measurement settings.

# Data

The raw micro-CT and segmented rock volumes are intentionally excluded from Git because they are large and originate from an external research-data provider.

The thesis notebooks use three-dimensional sandstone images and corresponding multimineral segmentations, including Bentheimer and Leopard sandstone volumes. The original research files reference Digital Rocks Portal project 305:

- Project page: <https://www.digitalrocksportal.org/projects/305>
- Portal home: <https://www.digitalrocksportal.org/>

Download data from the authoritative portal, retain its original metadata and license, and cite the dataset creators listed there.

## Expected local layout

```text
data/
├── raw/
│   ├── bentheimer/
│   │   ├── tomoHiRes_SS_nc/
│   │   │   ├── block00000000.nc
│   │   │   └── block00000001.nc
│   │   └── BHG1eff_tst_phase.nc
│   ├── leopard/
│   │   ├── tomo_R_SSw_SS_nc/
│   │   │   ├── block00000000.nc
│   │   │   └── block00000001.nc
│   │   └── LP_seg_800.nc
│   └── digital-rocks-portal-project-305/
└── processed/
    ├── unet/
    └── vox2vox/
```

Some notebooks create patches, masks, or binary volumes under `data/processed/`. These derived files are ignored by Git and should be regenerated locally.

Filenames can differ across portal downloads. If so, update only the corresponding repository-relative path and record the mapping in the preparation metadata.

## Data split recorded in the thesis

For the 800 × 800 × 800 volumes, the first 128 slices were used for validation, the next 128 for testing, and the remaining 544 for training. Patch sizes vary by experiment, principally 64³ and 128³ voxels. Confirm class labels and orientation against the portal metadata before training.

## Integrity and provenance

Record the portal project identifier, download date, filenames, checksums, voxel size, class mapping, and any preprocessing applied. Do not silently relabel, resample, or redistribute source volumes.

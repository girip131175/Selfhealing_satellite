# Reproducibility and External Artifacts

This repository contains the research source code for the Self-Healing Satellite system.

Large datasets, trained checkpoints, and generated outputs are intentionally not duplicated in the lightweight source distribution. The original research repository used Git LFS for these artifacts.

## Expected artifact locations

The code uses project-relative defaults and environment variables instead of machine-specific absolute paths.

Relevant environment variables include:

- `SELF_HEALING_GAN_CHECKPOINT`
- `SELF_HEALING_DAMAGE_CHECKPOINT`
- `SELF_HEALING_METHOD_CHECKPOINT`
- `SELF_HEALING_EDGE_DATASET`
- `SELF_HEALING_INTERFERENCE_DATA`
- `SELF_HEALING_INTERFERENCE_OUTPUT`
- `SELF_HEALING_INTERFERENCE_TEST_FILE`
- `SELF_HEALING_SPECTRUM_MODEL`
- `SELF_HEALING_INTERFERENCE_MODEL`
- `SELF_HEALING_GPT2_MODEL`
- `SELF_HEALING_REPORTS_DIR`
- `SELF_HEALING_SPECTRUM_PREDICTIONS`
- `SELF_HEALING_INTERFERENCE_DATASET`
- `SELF_HEALING_SPECTRUM_DATA_DIR`

If a required model or dataset is not present, the relevant training/evaluation pipeline should report the missing artifact rather than relying on a developer-specific filesystem path.

## Source integrity

The research algorithms were retained; this cleanup changes path/configuration handling for portability and public-source hygiene.

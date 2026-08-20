# Research Source Layout

The public repository separates the research implementation from large/generated artifacts.

## Intended source areas

- `spectrum/` — demand generation, forecasting, interference classification, and allocation experiments
- `self_healing/` — satellite damage detection, classification, restoration, and healing-strategy mapping
- `decision_support/` — report generation and language-model support components
- `evaluation/` — experiment/evaluation scripts
- `tests/` — lightweight validation code
- `data/` — metadata/configuration only; large datasets are not committed by default
- `models/` — model-loading interfaces/metadata; large weights are not committed by default

The original research code is preserved as provenance while public-facing source paths are progressively normalized to remove machine-specific filesystem assumptions.

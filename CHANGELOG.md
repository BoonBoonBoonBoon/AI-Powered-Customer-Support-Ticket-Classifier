# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Initial CI pipeline, Dockerization, structured logging, health endpoints, request size limiting.

## [1.0.6] - 2025-10-07
### Fixed
- Resolved v1.0.5 class collapse by disabling calibration.
- Added per-class recall gating (threshold 0.05) to training pipeline.
### Changed
- Reverted to raw LogisticRegression probabilities (no CalibratedClassifierCV).
### Metrics (Validation)
- Priority macro F1: 0.2494 (up from 0.1585 in 1.0.5, +0.091).
- Department macro F1: 0.3178 (up from 0.2545 in 1.0.5, +0.063).

## [1.0.5] - 2025-10-07
### Added
- Probability calibration (sigmoid) and calibration metrics persistence.
- Per-target C hyperparameters.
- Macro F1 gating script for CI.
### Regression
- Severe prediction collapse (department majority only, priority skew). Marked as non-promotable.

## [1.0.4] - 2025-10-07
### Added
- Engineered priority feature flag (`--priority-extra`) experimental (no net gain, left disabled by default).
### Improved
- Documentation and algorithm comparison groundwork.

## [1.0.3] - 2025-10-07
### Fixed
- Department leakage via `__type_*` token exclusion regex.
### Added
- Split index persistence and leakage guard warning.

## [1.0.0] - Initial model release
- Baseline classifier, training pipeline, versioned artifacts.

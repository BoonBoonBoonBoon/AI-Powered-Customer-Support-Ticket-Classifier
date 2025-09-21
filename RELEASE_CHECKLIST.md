# Release Checklist

## Pre-Release
- [ ] All tests passing in CI
- [ ] Lint & type checks clean (or justified)
- [ ] Security scan (pip-audit) reviewed
- [ ] Model re-trained (if required) with updated data
- [ ] `model_metadata.json` updated & validated
- [ ] Changelog updated with new version section
- [ ] Version bump applied in `app/config.py` and docs
- [ ] Docker image builds locally

## Release Tagging
- [ ] Create git tag `vX.Y.Z`
- [ ] CI publishes image to registry

## Post-Release
- [ ] Deploy to staging environment
- [ ] Smoke test `/version`, `/health/ready`, `/classify`
- [ ] Monitor logs & metrics for anomalies
- [ ] Update any external documentation portals

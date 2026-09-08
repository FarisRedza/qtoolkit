# qtoolkit

A Python package for quantum optics and data analysis

Current functionality:
- Timetag correlation for two-, three-, and four-fold coincidences
- Simulating timetag data
- Convenience tools for handling timetag data
- Functions for calculating common experimental qunatities such as QBER, visibility, fidelity, purity etc.

# Documentation
```bash
make -C docs clean && make -C docs dirhtml
```

# Testing
```bash
pytest
```
```bash
nox
```
```bash
act push -W .github/workflows/test_main.yaml
act -e .act/tag-push.json
```

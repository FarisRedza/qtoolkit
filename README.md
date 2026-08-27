# qtoolkit

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

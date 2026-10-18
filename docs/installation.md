# Installation for static inspection

Use a new environment, never overwrite an existing research environment. From the extracted repository root:

```shell
python -m venv .venv-static
```

Windows PowerShell activation: `.venv-static/Scripts/Activate.ps1`. POSIX shell activation: `source .venv-static/bin/activate`. Alternatively call the environment's Python executable directly if shell activation is restricted.

```shell
python -m pip install -r requirements-static.txt
python scripts/show_results.py
python scripts/validate_release.py
```

The results viewer needs only the standard library; the static validator also needs PyYAML and nbformat. No torch, MRI, model import or notebook execution is needed. Validation does not collect arbitrary tests. `scripts/phase6_smoke.py` is an experimental historical helper and is excluded from this procedure.

Preserve the src.* namespace and work from the repository root. Do not use `pip install -e .`; pyproject.toml is tool configuration, not a new packaging interface. The historical environment files remain evidence, not a supported cross-platform installer. See [environment compatibility](environment_compatibility.md) for the actual installation verification performed.

For future authorized research, use a separate private workspace. MORPHONET_ROOT points to that experiment root; MORPHONET_DATA_ROOT controls dataset discovery only. Environment variables must be set explicitly; no dotenv loading is added. Existing frozen split paths and fingerprints cannot be rewritten by these variables. Source integrity checks remain fail-closed; never bypass them to claim compatibility.

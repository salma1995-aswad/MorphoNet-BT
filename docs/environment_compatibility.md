# Environment compatibility and verification scope

| Environment | Evidence and intended use | Status |
|---|---|---|
| Historical research | requirements.txt and environment.yml | Recorded versions; not a fresh cross-platform solved lock |
| Static inspection | Python, PyYAML 6.0.3, nbformat 5.10.4 and their resolved dependencies | Installation outcome recorded in release audit |
| Research execution | Compatible torch/torchvision, CUDA/driver, scientific dependencies, private artifacts and original integrity records | Not installed or experimentally verified in Phase 18.2 |

Historical observations include Python 3.11.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA 13.0 and an RTX 3070 Laptop GPU. These labels are preserved as reported evidence; they are not a claim that every pin has a publicly available wheel. The original requirements are not a complete transitive lock. No current GPU probe was performed. Consult official PyPI metadata and the official PyTorch installation selector before a future authorized installation; do not silently substitute versions and call the result equivalent.

The minimal static environment intentionally uses a separate nbformat version from the historical environment. This affects release inspection only and does not establish research-runtime equivalence. The release audit states whether its clean isolated installation succeeded. No existing dl environment was changed. No notebook or model was run to test compatibility. Windows historical environment-name assertions and AMP behavior remain unchanged; CPU and other-platform research execution are unverified.


## Phase 18.2 verification outcome

A fresh isolated Windows x64 Python 3.12.14 environment successfully installed the minimal static requirements and ran all 14 static tests. `pip check` passed. Standard venv creation could not complete ensurepip in this host; an existing pip executable was explicitly targeted at the new venv to bootstrap pip and dependencies. No system-site packages were enabled and the research environment was not altered. This host-specific bootstrap is not presented as evidence that venv creation works on every platform.

Official metadata confirmed available wheels for [PyYAML 6.0.3](https://pypi.org/pypi/PyYAML/6.0.3/json) and [nbformat 5.10.4](https://pypi.org/pypi/nbformat/5.10.4/json). Matching historical CUDA-tagged wheel names were found in the official [torch index](https://download.pytorch.org/whl/cu130/torch/) and [torchvision index](https://download.pytorch.org/whl/cu130/torchvision/). Existence is not installation, driver compatibility or experimental validation. Other historical pins were preserved but not independently resolved. Detailed installation records remain in the internal release reports; they are not required by the public static checks. The duplicate historical requirements file has been removed; requirements.txt retains the exact observed research pins and environment.yml still references it.

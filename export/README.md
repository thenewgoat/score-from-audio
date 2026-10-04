# export

Turns the two models into the ONNX graphs the app runs, and proves each graph
equal to its original. Output goes to `android/models/`, which is gitignored.

| step | needs | command |
|---|---|---|
| Transkun graphs (fp32 core, int8 core for the app) | the `transkun` pip package | `python -m export.transkun_export` |
| Transkun fixtures | the `transkun` pip package | `python -m export.transkun_fixtures` |
| bd1 graphs (`--checkpoint`, a bd1 `.pt`) | torch | `python -m export.bd1_export --checkpoint bd1.pt` |

Tests: `python -m pytest export/tests` (with `transkun` installed).
Paths are in `export/paths.py`; each can be overridden by environment variable.

bd1's checkpoint and its training code are not in this repository; the
exported graphs ship inside the APK.

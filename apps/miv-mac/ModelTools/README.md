# MIV model tools

`Snapshot/omlx` is an editable Apache-2.0 source snapshot of the download,
upload, oQ quantization, calibration, model-family support and client integration
modules from this repository at `338e927943dd7b75880092c5fb5797922a1c8c9a`.
The import namespace is retained for compatibility with the quantizer's model
patches. The integration launchers register a separate `miv` provider.

`stage.sh` copies this directory's tools into the app's ModelTools resources.
The app also bundles its own CPython, MLX, Hugging Face and ModelScope libraries.
The model-tool worker accepts JSON over its parent's pipes. Jobs continue while
the user changes pages and shut down on EOF. Downloads and quantization outputs
are written to the MIV model library.

Third-party clients use an authenticated loopback relay. Each client launch
profile contains a temporary relay credential and model selection. The upstream
inference key stays in worker memory. The relay forwards OpenAI Chat and
translates Responses / Anthropic Messages, including SSE and function calls,
to the selected MIV model.

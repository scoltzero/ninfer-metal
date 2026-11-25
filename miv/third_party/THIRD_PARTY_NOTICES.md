# Third-Party Notices (`miv/third_party/` & Adapted Sources)

> **Project License Note:** `ninfer-metal` (`miv/`, `omlx-extend/`) is licensed under **GNU GPL-3.0** (see root [`LICENSE`](../../LICENSE)).
> Upstream vendored `omlx/` retains its original Apache-2.0 notice ([`omlx/LICENSE`](../../omlx/LICENSE)), and the third-party components below retain their respective permissive licenses and notices.

This directory contains vendored third-party libraries used by `ninfer-metal` (`miv`), including the standalone native HTTP server `ninfer-metal-serve`. Each library retains its original copyright and license file in its respective subdirectory.

## Vendored Libraries

### 1. nlohmann/json
- **Path**: `miv/third_party/nlohmann/`
- **License**: MIT License (`miv/third_party/nlohmann/LICENSE.MIT`)
- **Copyright**: Copyright (c) 2013-2025 Niels Lohmann

### 2. cpp-httplib
- **Path**: `miv/third_party/cpp-httplib/`
- **License**: MIT License (`miv/third_party/cpp-httplib/LICENSE`)
- **Copyright**: Copyright (c) 2017 yhirose / Yuji Hirose
- **Notes**: Vendored from `ninfer/third_party/cpp-httplib` (header-only POSIX socket build, no external TLS/compression runtime dependencies).

### 3. toml++ (tomlplusplus)
- **Path**: `miv/third_party/tomlplusplus/`
- **License**: MIT License (`miv/third_party/tomlplusplus/LICENSE`)
- **Copyright**: Copyright (c) Mark Gillard

### 4. utf8proc
- **Path**: `miv/third_party/utf8proc/`
- **License**: MIT License + Unicode Data License (`miv/third_party/utf8proc/LICENSE.md`)
- **Copyright**:
  - Copyright (c) 2014-2021 Steven G. Johnson, Jiahao Chen, Tony Kelman, Jonas Fonseca, and contributors
  - Copyright (c) 2009, 2013 Public Software Group e. V., Berlin, Germany
  - Copyright (c) 1991-2007 Unicode, Inc. (Unicode data in `utf8proc_data.c`)

### 5. llama-jinja (NInfer fork of llama.cpp `common/jinja`)
- **Path**: `miv/third_party/llama-jinja/`
- **License**: MIT License (`miv/third_party/llama-jinja/LICENSE`) + Unicode License v3 (`miv/third_party/llama-jinja/UNICODE-LICENSE`)
- **Copyright**:
  - Copyright (c) 2023-2026 The ggml authors
  - Copyright (c) 1991-2026 Unicode, Inc.
- **Modifications**: Vendored from `ninfer/third_party/llama-jinja` (forked from `llama.cpp` commit `7609846557c50f9d984719a9e1e8c5f3d02f807b`); adapted `jinja/string.cpp` and `jinja/unicode.cpp` include path (`serve/unicode.h`) and namespace (`omlx::m5::serve::unicode_internal`) for `miv`.

### 6. TensorFold
- **Path**: `miv/third_party/tensorfold/`
- **License**: Apache-2.0 (`LICENSE-APACHE-2.0`) / MIT (`LICENSE-MIT`) + `NOTICE`

## Adapted First-Party / Apache-2.0 Sources

The following files in `miv/src/serve/` are adapted from `ninfer/src/text/` (Apache License, Version 2.0; see repository `ninfer/LICENSE`), retaining Apache-2.0 origin and modification notices in each file header:
- `miv/src/serve/byte_span.h` (adapted from `ninfer/src/text/byte_span.h`)
- `miv/src/serve/unicode.h`, `miv/src/serve/unicode.cpp` (adapted from `ninfer/src/text/unicode.{h,cpp}`)
- `miv/src/serve/tokenizer.h`, `miv/src/serve/tokenizer.cpp` (adapted from `ninfer/src/text/tokenizer.{h,cpp}`, adding `BpeSplit::LetterDigit1`, optional `tokenizer_config_json`, and token-ID lookup helper)
- `miv/src/serve/jinja.h`, `miv/src/serve/jinja.cpp` (adapted from `ninfer/src/text/jinja.{h,cpp}`)

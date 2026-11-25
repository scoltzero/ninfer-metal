// Copyright 2026 The NInfer Authors / omlx-backend-m5ultra-ninfer contributors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Adapted from ninfer/src/text/jinja.h (Apache-2.0).
// Modifications: moved into namespace omlx::m5::serve and updated include path.

#pragma once

#include "serve/byte_span.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace omlx::m5::serve {

struct TemplateInputRegion {
  std::string pointer;
  std::uint32_t tag = 0;
};

struct TemplateOutputRegion {
  std::uint32_t tag = 0;
  std::size_t begin = 0;
  std::size_t end = 0;
  std::optional<std::size_t> source_offset;
};

struct TemplateBoundaryMapping {
  std::uint32_t tag = 0;
  std::size_t source_begin = 0;
  std::size_t source_end = 0;
  std::size_t offset = 0;
};

struct TemplateRenderOptions {
  std::time_t timestamp = std::time(nullptr);
  std::function<void()> checkpoint;
  std::span<const TemplateInputRegion> regions;
  std::span<const std::string> control_variables;
};

struct TemplateOutput {
  std::string text;
  std::vector<ByteSpan> literal_spans;
  std::vector<TemplateOutputRegion> regions;
  std::vector<TemplateBoundaryMapping> boundary_mappings;
};

class JinjaTemplate {
public:
  explicit JinjaTemplate(std::string source, std::string source_name);
  TemplateOutput render(const nlohmann::ordered_json &context,
                        const TemplateRenderOptions &options = {}) const;

private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

} // namespace omlx::m5::serve

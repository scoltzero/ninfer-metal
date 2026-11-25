#pragma once

#include "serve/chat_template.h"
#include "serve/config.h"
#include "serve/engine_router.h"
#include "serve/multi_model_host.h"
#include "serve/tokenizer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace omlx::m5::serve {

class ServeHttpServer final {
public:
  // Backward-compatible single-model constructor.
  ServeHttpServer(ServeConfig config, std::shared_ptr<EngineRouter> router,
                  std::shared_ptr<Tokenizer> tokenizer,
                  std::shared_ptr<ChatTemplate> chat_template);

  // Multi-model parallel host constructor.
  ServeHttpServer(ServeConfig config, std::shared_ptr<MultiModelHost> host);

  ~ServeHttpServer();

  ServeHttpServer(const ServeHttpServer &) = delete;
  ServeHttpServer &operator=(const ServeHttpServer &) = delete;

  // Binds the TCP listener (port == 0 picks an ephemeral port) and returns the
  // actual bound port.
  int bind();

  // Blocks serving requests until stop() is called.
  void listen_after_bind();

  // Thread-safe / signal-handler-friendly server stop trigger.
  void stop();

  [[nodiscard]] int bound_port() const noexcept;
  [[nodiscard]] const std::string &model_id() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5::serve

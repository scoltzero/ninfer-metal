#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5 {

enum class TensorDType {
  u8,
  u16,
  u32,
  i8,
  i16,
  i32,
  i64,
  f16,
  bf16,
  f32,
  f64,
  boolean,
};

struct TensorInfo {
  std::string name;
  std::string shard;
  TensorDType dtype = TensorDType::u8;
  std::vector<std::uint64_t> shape;
  std::uint64_t data_offset = 0;
  std::uint64_t nbytes = 0;
};

struct TensorView {
  const std::byte *data = nullptr;
  std::uint64_t nbytes = 0;
  const TensorInfo *info = nullptr;
};

class SafeTensorModel final {
public:
  static SafeTensorModel open(const std::filesystem::path &model_dir,
                              std::string_view expected_revision = {});
  static SafeTensorModel open_file(const std::filesystem::path &file_path);

  SafeTensorModel(SafeTensorModel &&) noexcept;
  SafeTensorModel &operator=(SafeTensorModel &&) noexcept;
  SafeTensorModel(const SafeTensorModel &) = delete;
  SafeTensorModel &operator=(const SafeTensorModel &) = delete;
  ~SafeTensorModel();

  [[nodiscard]] const std::filesystem::path &model_dir() const;
  [[nodiscard]] const std::string &revision() const;
  [[nodiscard]] std::uint64_t declared_total_size() const;
  [[nodiscard]] std::uint64_t total_tensor_bytes() const;
  [[nodiscard]] std::size_t shard_count() const;
  [[nodiscard]] std::size_t tensor_count() const;
  [[nodiscard]] std::vector<std::string> tensor_names() const;
  [[nodiscard]] const TensorInfo &tensor_info(std::string_view name) const;
  [[nodiscard]] TensorView tensor(std::string_view name) const;

private:
  class Impl;

  explicit SafeTensorModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::size_t dtype_size(TensorDType dtype);
[[nodiscard]] std::string_view dtype_name(TensorDType dtype);

} // namespace omlx::m5

#include "omlx_m5/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace omlx::m5 {

namespace {

using Json = nlohmann::json;

class MappedFile {
public:
  explicit MappedFile(const std::filesystem::path &path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("unable to open " + path.string());
    }
    struct stat status{};
    if (::fstat(fd_, &status) != 0 || status.st_size <= 0) {
      close_file();
      throw std::runtime_error("unable to stat non-empty file " +
                               path.string());
    }
    size_ = static_cast<std::uint64_t>(status.st_size);
    mapping_ = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ,
                      MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      close_file();
      throw std::runtime_error("unable to mmap " + path.string());
    }
  }

  MappedFile(MappedFile &&other) noexcept
      : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)),
        mapping_(std::exchange(other.mapping_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      reset();
      path_ = std::move(other.path_);
      fd_ = std::exchange(other.fd_, -1);
      mapping_ = std::exchange(other.mapping_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() { reset(); }

  [[nodiscard]] const std::byte *data() const {
    return static_cast<const std::byte *>(mapping_);
  }
  [[nodiscard]] std::uint64_t size() const { return size_; }

private:
  void close_file() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  void reset() {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, static_cast<std::size_t>(size_));
      mapping_ = nullptr;
    }
    close_file();
    size_ = 0;
  }

  std::filesystem::path path_;
  int fd_ = -1;
  void *mapping_ = nullptr;
  std::uint64_t size_ = 0;
};

TensorDType parse_dtype(std::string_view value) {
  static const std::map<std::string_view, TensorDType> types = {
      {"U8", TensorDType::u8},     {"U16", TensorDType::u16},
      {"U32", TensorDType::u32},   {"I8", TensorDType::i8},
      {"I16", TensorDType::i16},   {"I32", TensorDType::i32},
      {"I64", TensorDType::i64},   {"F16", TensorDType::f16},
      {"BF16", TensorDType::bf16}, {"F32", TensorDType::f32},
      {"F64", TensorDType::f64},   {"BOOL", TensorDType::boolean},
  };
  const auto it = types.find(value);
  if (it == types.end()) {
    throw std::runtime_error("unsupported safetensors dtype: " +
                             std::string(value));
  }
  return it->second;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error("tensor element count overflow");
  }
  return left * right;
}

std::uint64_t expected_tensor_bytes(const std::vector<std::uint64_t> &shape,
                                    TensorDType dtype) {
  std::uint64_t elements = 1;
  for (const auto dimension : shape) {
    elements = checked_multiply(elements, dimension);
  }
  return checked_multiply(elements, dtype_size(dtype));
}

Json parse_json_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open JSON file " + path.string());
  }
  try {
    return Json::parse(input);
  } catch (const Json::exception &error) {
    throw std::runtime_error("invalid JSON in " + path.string() + ": " +
                             error.what());
  }
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open " + path.string());
  }
  std::string value;
  std::getline(input, value);
  if (value.empty()) {
    throw std::runtime_error("empty file " + path.string());
  }
  return value;
}

struct Shard {
  explicit Shard(const std::filesystem::path &path) : file(path) {
    if (file.size() < sizeof(std::uint64_t)) {
      throw std::runtime_error("safetensors file is shorter than its prefix");
    }
    std::uint64_t header_bytes = 0;
    std::memcpy(&header_bytes, file.data(), sizeof(header_bytes));
    if constexpr (std::endian::native == std::endian::big) {
      header_bytes = __builtin_bswap64(header_bytes);
    }
    constexpr std::uint64_t max_header_bytes = 128ULL * 1024ULL * 1024ULL;
    if (header_bytes == 0 || header_bytes > max_header_bytes ||
        header_bytes > file.size() - sizeof(std::uint64_t)) {
      throw std::runtime_error("invalid safetensors header length");
    }
    data_start = sizeof(std::uint64_t) + header_bytes;
    const auto *header_begin =
        reinterpret_cast<const char *>(file.data() + sizeof(std::uint64_t));
    Json header;
    try {
      header = Json::parse(header_begin, header_begin + header_bytes);
    } catch (const Json::exception &error) {
      throw std::runtime_error(
          std::string("invalid safetensors header JSON: ") + error.what());
    }
    if (!header.is_object()) {
      throw std::runtime_error("safetensors header must be an object");
    }
    for (const auto &[name, value] : header.items()) {
      if (name == "__metadata__") {
        continue;
      }
      if (!value.is_object() || !value.contains("dtype") ||
          !value.contains("shape") || !value.contains("data_offsets")) {
        throw std::runtime_error("incomplete tensor record: " + name);
      }
      TensorInfo info;
      info.name = name;
      info.dtype = parse_dtype(value.at("dtype").get<std::string>());
      info.shape = value.at("shape").get<std::vector<std::uint64_t>>();
      const auto offsets =
          value.at("data_offsets").get<std::vector<std::uint64_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0]) {
        throw std::runtime_error("invalid tensor offsets: " + name);
      }
      info.data_offset = data_start + offsets[0];
      info.nbytes = offsets[1] - offsets[0];
      if (offsets[1] > file.size() - data_start) {
        throw std::runtime_error("tensor extends beyond shard: " + name);
      }
      const auto expected = expected_tensor_bytes(info.shape, info.dtype);
      if (expected != info.nbytes) {
        throw std::runtime_error(
            "tensor byte count does not match shape/dtype: " + name);
      }
      const auto [_, inserted] = tensors.emplace(name, std::move(info));
      if (!inserted) {
        throw std::runtime_error("duplicate tensor in shard: " + name);
      }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(tensors.size());
    for (const auto &[_, info] : tensors) {
      ranges.emplace_back(info.data_offset, info.data_offset + info.nbytes);
    }
    std::ranges::sort(ranges);
    std::uint64_t previous_end = data_start;
    for (const auto &[begin, end] : ranges) {
      if (begin < previous_end) {
        throw std::runtime_error("safetensors tensor payloads overlap");
      }
      previous_end = end;
    }
  }

  MappedFile file;
  std::uint64_t data_start = 0;
  std::unordered_map<std::string, TensorInfo> tensors;
};

void validate_shard_name(std::string_view value) {
  const std::filesystem::path path(value);
  if (path.empty() || path.is_absolute() || path.filename() != path) {
    throw std::runtime_error("unsafe shard path in index: " +
                             std::string(value));
  }
}

} // namespace

class SafeTensorModel::Impl {
public:
  struct Locator {
    TensorInfo info;
    std::size_t shard_index = 0;
  };

  std::filesystem::path model_dir;
  std::string revision;
  std::uint64_t declared_total_size = 0;
  std::uint64_t total_shard_bytes = 0;
  std::uint64_t total_tensor_bytes = 0;
  std::vector<Shard> shards;
  std::unordered_map<std::string, Locator> tensors;
};

SafeTensorModel SafeTensorModel::open(const std::filesystem::path &model_dir,
                                      std::string_view expected_revision) {
  const auto canonical_dir = std::filesystem::weakly_canonical(model_dir);
  if (!std::filesystem::is_directory(canonical_dir)) {
    throw std::runtime_error("model directory does not exist: " +
                             model_dir.string());
  }
  auto impl = std::make_unique<Impl>();
  impl->model_dir = canonical_dir;
  const auto revision_path = canonical_dir / "REVISION";
  if (std::filesystem::is_regular_file(revision_path)) {
    impl->revision = read_text_file(revision_path);
  } else if (!expected_revision.empty()) {
    throw std::runtime_error("model revision file is missing: " +
                             revision_path.string());
  }
  if (!expected_revision.empty() && impl->revision != expected_revision) {
    throw std::runtime_error("model revision mismatch: expected " +
                             std::string(expected_revision) + ", found " +
                             impl->revision);
  }

  const auto index_path = canonical_dir / "model.safetensors.index.json";
  const auto index = parse_json_file(index_path);
  if (!index.contains("metadata") || !index.contains("weight_map")) {
    throw std::runtime_error(
        "safetensors index is missing metadata or weight_map");
  }
  impl->declared_total_size =
      index.at("metadata").at("total_size").get<std::uint64_t>();
  const auto &weight_map = index.at("weight_map");
  if (!weight_map.is_object() || weight_map.empty()) {
    throw std::runtime_error(
        "safetensors weight_map must be a non-empty object");
  }

  std::map<std::string, std::size_t> shard_indices;
  for (const auto &[_, shard_value] : weight_map.items()) {
    const auto shard_name = shard_value.get<std::string>();
    validate_shard_name(shard_name);
    if (!shard_indices.contains(shard_name)) {
      const auto shard_index = impl->shards.size();
      shard_indices.emplace(shard_name, shard_index);
      impl->shards.emplace_back(canonical_dir / shard_name);
      impl->total_shard_bytes += impl->shards.back().file.size();
    }
  }

  for (const auto &[tensor_name, shard_value] : weight_map.items()) {
    const auto shard_name = shard_value.get<std::string>();
    const auto shard_index = shard_indices.at(shard_name);
    const auto &shard = impl->shards.at(shard_index);
    const auto tensor_it = shard.tensors.find(tensor_name);
    if (tensor_it == shard.tensors.end()) {
      throw std::runtime_error(
          "index maps tensor to a shard that does not contain it: " +
          tensor_name);
    }
    auto info = tensor_it->second;
    info.shard = shard_name;
    impl->total_tensor_bytes += info.nbytes;
    impl->tensors.emplace(
        tensor_name,
        Impl::Locator{.info = std::move(info), .shard_index = shard_index});
  }

  std::size_t shard_tensor_count = 0;
  for (const auto &shard : impl->shards) {
    shard_tensor_count += shard.tensors.size();
  }
  if (shard_tensor_count != impl->tensors.size()) {
    throw std::runtime_error(
        "safetensors index omits or duplicates tensors from its shards");
  }
  // Hugging Face's index convention defines metadata.total_size as tensor
  // payload bytes. The fixed oQ4e artifact inherited a converter convention
  // that stores physical shard bytes instead. Both are unambiguous after the
  // complete index/shard audit above, so accept exactly those two meanings.
  if (impl->declared_total_size != impl->total_shard_bytes &&
      impl->declared_total_size != impl->total_tensor_bytes) {
    throw std::runtime_error(
        "safetensors index total_size matches neither tensor nor shard bytes");
  }
  return SafeTensorModel(std::move(impl));
}

SafeTensorModel
SafeTensorModel::open_file(const std::filesystem::path &file_path) {
  const auto canonical_file = std::filesystem::weakly_canonical(file_path);
  if (!std::filesystem::is_regular_file(canonical_file)) {
    throw std::runtime_error("safetensors file does not exist: " +
                             file_path.string());
  }
  auto impl = std::make_unique<Impl>();
  impl->model_dir = canonical_file.parent_path();
  const auto shard_name = canonical_file.filename().string();
  impl->shards.emplace_back(canonical_file);
  const auto &shard = impl->shards.back();
  impl->total_shard_bytes = shard.file.size();
  for (const auto &[tensor_name, raw_info] : shard.tensors) {
    auto info = raw_info;
    info.shard = shard_name;
    impl->total_tensor_bytes += info.nbytes;
    impl->tensors.emplace(
        tensor_name,
        Impl::Locator{.info = std::move(info), .shard_index = 0});
  }
  impl->declared_total_size = impl->total_tensor_bytes;
  return SafeTensorModel(std::move(impl));
}

SafeTensorModel::SafeTensorModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SafeTensorModel::SafeTensorModel(SafeTensorModel &&) noexcept = default;
SafeTensorModel &
SafeTensorModel::operator=(SafeTensorModel &&) noexcept = default;
SafeTensorModel::~SafeTensorModel() = default;

const std::filesystem::path &SafeTensorModel::model_dir() const {
  return impl_->model_dir;
}
const std::string &SafeTensorModel::revision() const { return impl_->revision; }
std::uint64_t SafeTensorModel::declared_total_size() const {
  return impl_->declared_total_size;
}
std::uint64_t SafeTensorModel::total_tensor_bytes() const {
  return impl_->total_tensor_bytes;
}
std::size_t SafeTensorModel::shard_count() const {
  return impl_->shards.size();
}
std::size_t SafeTensorModel::tensor_count() const {
  return impl_->tensors.size();
}

std::vector<std::string> SafeTensorModel::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(impl_->tensors.size());
  for (const auto &[name, _] : impl_->tensors) {
    names.push_back(name);
  }
  std::ranges::sort(names);
  return names;
}

const TensorInfo &SafeTensorModel::tensor_info(std::string_view name) const {
  const auto it = impl_->tensors.find(std::string(name));
  if (it == impl_->tensors.end()) {
    throw std::out_of_range("unknown tensor: " + std::string(name));
  }
  return it->second.info;
}

TensorView SafeTensorModel::tensor(std::string_view name) const {
  const auto it = impl_->tensors.find(std::string(name));
  if (it == impl_->tensors.end()) {
    throw std::out_of_range("unknown tensor: " + std::string(name));
  }
  const auto &locator = it->second;
  const auto &shard = impl_->shards.at(locator.shard_index);
  return {
      .data = shard.file.data() + locator.info.data_offset,
      .nbytes = locator.info.nbytes,
      .info = &locator.info,
  };
}

std::size_t dtype_size(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::u8:
  case TensorDType::i8:
  case TensorDType::boolean:
    return 1;
  case TensorDType::u16:
  case TensorDType::i16:
  case TensorDType::f16:
  case TensorDType::bf16:
    return 2;
  case TensorDType::u32:
  case TensorDType::i32:
  case TensorDType::f32:
    return 4;
  case TensorDType::i64:
  case TensorDType::f64:
    return 8;
  }
  throw std::logic_error("unhandled tensor dtype");
}

std::string_view dtype_name(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::u8:
    return "U8";
  case TensorDType::u16:
    return "U16";
  case TensorDType::u32:
    return "U32";
  case TensorDType::i8:
    return "I8";
  case TensorDType::i16:
    return "I16";
  case TensorDType::i32:
    return "I32";
  case TensorDType::i64:
    return "I64";
  case TensorDType::f16:
    return "F16";
  case TensorDType::bf16:
    return "BF16";
  case TensorDType::f32:
    return "F32";
  case TensorDType::f64:
    return "F64";
  case TensorDType::boolean:
    return "BOOL";
  }
  throw std::logic_error("unhandled tensor dtype");
}

} // namespace omlx::m5

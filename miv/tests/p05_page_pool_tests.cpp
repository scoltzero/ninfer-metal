#include "omlx_m5/kv_page_pool.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace omlx::m5;
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<std::uint32_t> table(const KvPagePool &pool) {
  auto bytes = pool.table_buffer().read();
  std::vector<std::uint32_t> values(bytes.size() / 4);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("p05_page_pool_tests LIBRARIES");
    auto context = MetalContext::create();
    KvPagePool pool(context, 12, 4, 4);
    require(pool.logical_pages() == 3 && pool.physical_pages() == 4 &&
                pool.free_pages() == 4 && pool.mapped_pages() == 0,
            "P05 initial pool geometry");
    for (auto width : {3U, 1U, 1U, 4U}) {
      const auto moves = pool.begin_append(width);
      require(moves.empty(), "P05 unexpected COW on sole owner");
      pool.commit(width);
    }
    require(pool.tokens() == 9 && pool.mapped_pages() == 3 &&
                table(pool) == std::vector<std::uint32_t>({0, 1, 2}),
            "P05 P-1/P/P+1 mapping");
    pool.truncate(4);
    require(pool.mapped_pages() == 1 && pool.free_pages() == 3,
            "P05 truncate did not free tail pages");
    static_cast<void>(pool.begin_append(8));
    pool.commit(8);
    require(pool.tokens() == 12 &&
                table(pool) == std::vector<std::uint32_t>({0, 2, 1}),
            "P05 physical page reuse did not permute mapping");
    const auto lease = pool.retain_prefix();
    pool.truncate(10);
    const auto moves = pool.begin_append(1);
    require(moves.size() == 1 && moves[0].source == 1 &&
                moves[0].destination == 3 && pool.physical_slot(2) == 3,
            "P05 shared tail did not COW");
    auto keys = context.make_buffer(4ULL * 4 * 8 * 2);
    auto values = context.make_buffer(keys.size());
    std::vector<std::byte> page(4 * 8 * 2);
    for (std::size_t i = 0; i < page.size(); ++i)
      page[i] = std::byte((i * 17) & 0xff);
    keys.write(page, std::uint64_t(moves[0].source) * page.size());
    values.write(page, std::uint64_t(moves[0].source) * page.size());
    KvPageCopier copier(context, keys, values, 4, 8, 4, argv[1], "test.cow");
    static_cast<void>(copier.execute(moves[0]));
    require(keys.read(std::uint64_t(moves[0].destination) * page.size(),
                      page.size()) == page &&
                values.read(std::uint64_t(moves[0].destination) * page.size(),
                            page.size()) == page,
            "P05 COW GPU page contents changed");
    keys.write(std::vector<std::byte>(page.size()),
               std::uint64_t(moves[0].destination) * page.size());
    require(keys.read(std::uint64_t(moves[0].source) * page.size(),
                      page.size()) == page,
            "P05 COW changed retained physical page");
    pool.commit(1);
    require(pool.tokens() == 11 && table(pool)[2] == 3,
            "P05 COW commit did not retain new tail");
    pool.release_prefix(lease);
    require(pool.free_pages() == 1, "P05 lease release leaked old page");
    pool.reset();
    require(pool.tokens() == 0 && pool.mapped_pages() == 0 &&
                pool.free_pages() == 4,
            "P05 reset leaked physical pages");
    bool rejected = false;
    try {
      pool.truncate(1);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "P05 truncate past frontier accepted");

    KvPagePool rollback(context, 10, 4, 4);
    static_cast<void>(rollback.begin_append(9));
    rollback.abort();
    require(rollback.tokens() == 0 && rollback.free_pages() == 4 &&
                table(rollback) ==
                    std::vector<std::uint32_t>(3, KvPagePool::unmapped),
            "P05 pending abort exposed pages");
    static_cast<void>(rollback.begin_append(3));
    rollback.commit(3);
    const auto before = table(rollback);
    static_cast<void>(rollback.begin_append(1));
    rollback.commit(0);
    require(rollback.tokens() == 3 && table(rollback) == before &&
                rollback.free_pages() == 3,
            "P05 zero-prefix commit did not roll back page ownership");
    const auto hold = rollback.retain_prefix();
    KvPagePool limited(context, 8, 4, 2);
    static_cast<void>(limited.begin_append(5));
    limited.commit(5);
    const auto limited_hold = limited.retain_prefix();
    rejected = false;
    try {
      static_cast<void>(limited.begin_append(1));
    } catch (const std::overflow_error &) {
      rejected = true;
    }
    require(rejected && limited.tokens() == 5 && limited.mapped_pages() == 2 &&
                table(limited) == std::vector<std::uint32_t>({0, 1}),
            "P05 OOM changed committed mapping");
    limited.release_prefix(limited_hold);
    rollback.release_prefix(hold);
    require(table(rollback) == before, "P05 lease release changed mapping");
    std::cout << "P05 page ownership, permutation, COW and rollback passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

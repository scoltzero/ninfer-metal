#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qwen_forward.h"

using namespace omlx::m5;
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<std::uint32_t> prompt(std::size_t count) {
  std::vector<std::uint32_t> ids(count);
  for (std::size_t i = 0; i < count; ++i)
    ids[i] = std::uint32_t((71 * i + 256) % 257);
  return ids;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("p05_qwen_paging_tests FIXTURES LIBRARIES");
    const auto root = std::filesystem::path(argv[1]) / "affine";
    auto config = load_qwen_model_config(root / "source/config.json");
    auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
    auto context = MetalContext::create();
    auto teacher_context = MetalContext::create();
    auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
    auto weights = materialize_model(
        artifact, plan_model_materialization(artifact, context), context);
    auto teacher_weights = materialize_model(
        artifact, plan_model_materialization(artifact, teacher_context),
        teacher_context);
    const std::array<std::uint32_t, 2> taps{0, 1};
    QwenForward program(context, weights, layers, config, 1024, argv[2], taps,
                        false, true);
    QwenForward teacher(teacher_context, teacher_weights, layers, config, 1024,
                        argv[2], taps, false, true);
    const auto long_prompt = prompt(129);
    const auto first = program.run(long_prompt, {}, true);
    const auto first_table = program.inspect_kv_page_table();
    require(first_table.size() == 16 && first_table[0] == 0 &&
                first_table[1] == 1 && first_table[2] == 2,
            "P05 first pages were not allocated on demand");
    program.reset();
    const auto reused = program.run(long_prompt, {}, true);
    const auto reused_table = program.inspect_kv_page_table();
    require(reused_table[0] == 2 && reused_table[1] == 1 &&
                reused_table[2] == 0 && first.hidden == reused.hidden &&
                first.logits == reused.logits && first.token == reused.token,
            "P05 noncontiguous physical page reuse changed forward result");

    program.reset();
    const auto boundary_prompt = prompt(63);
    static_cast<void>(program.run(boundary_prompt));
    static_cast<void>(teacher.run(boundary_prompt));
    const auto before = program.inspect_kv_page_table();
    const auto gdn_before = program.inspect_recurrent_state();
    const auto lease = program.diagnostic_retain_kv_pages();
    const std::array<std::uint32_t, 8> input{7, 3, 100, 200, 256, 44, 23, 87};
    static_cast<void>(program.verify(input));
    const auto tentative = program.inspect_kv_page_table();
    require(tentative[0] != before[0] && tentative[1] != UINT32_MAX &&
                program.position() == 63 &&
                program.inspect_recurrent_state() == gdn_before,
            "P05 COW verify exposed uncommitted state");
    program.abort_verify();
    require(program.inspect_kv_page_table() == before &&
                program.position() == 63,
            "P05 abort did not restore committed page mapping");
    const std::array<std::uint32_t, 1> continuation{37};
    auto after_abort = program.run(continuation, {}, true);
    auto teacher_after_abort = teacher.run(continuation, {}, true);
    require(after_abort.logits == teacher_after_abort.logits &&
                program.inspect_recurrent_state() ==
                    teacher.inspect_recurrent_state(),
            "P05 abort/continuation differed from teacher");
    program.diagnostic_release_kv_pages(lease);

    program.reset();
    teacher.reset();
    static_cast<void>(program.run(boundary_prompt));
    static_cast<void>(teacher.run(boundary_prompt));
    const auto retained = program.diagnostic_retain_kv_pages();
    const auto committed_before = program.inspect_kv_page_table();
    static_cast<void>(program.verify(input));
    static_cast<void>(program.commit_prefix(2));
    require(program.position() == 65 &&
                program.inspect_kv_page_table()[0] != committed_before[0] &&
                program.inspect_kv_page_table()[1] != UINT32_MAX,
            "P05 partial commit did not retain COW and cross-page append");
    static_cast<void>(teacher.run(std::span(input).first(2)));
    require(program.inspect_recurrent_state() ==
                teacher.inspect_recurrent_state(),
            "P05 partial ReplaySSM state differs");
    auto after_commit = program.run(continuation, {}, true);
    auto teacher_after_commit = teacher.run(continuation, {}, true);
    require(after_commit.logits == teacher_after_commit.logits,
            "P05 partial-page continuation differs from teacher");
    program.diagnostic_release_kv_pages(retained);

    auto limited_context = MetalContext::create();
    auto limited_teacher_context = MetalContext::create();
    auto limited_layers = build_qwen_layer_catalog(artifact, config, 128, 64);
    auto limited_weights = materialize_model(
        artifact, plan_model_materialization(artifact, limited_context),
        limited_context);
    auto limited_teacher_weights = materialize_model(
        artifact, plan_model_materialization(artifact, limited_teacher_context),
        limited_teacher_context);
    QwenForward limited(limited_context, limited_weights, limited_layers,
                        config, 128, argv[2], taps, false, true);
    QwenForward limited_teacher(limited_teacher_context,
                                limited_teacher_weights, limited_layers, config,
                                128, argv[2], taps, false, true);
    const auto full_pages = prompt(65);
    static_cast<void>(limited.run(full_pages));
    static_cast<void>(limited_teacher.run(full_pages));
    const auto old_lease = limited.diagnostic_retain_kv_pages();
    static_cast<void>(limited.run(continuation));
    static_cast<void>(limited_teacher.run(continuation));
    const auto current_lease = limited.diagnostic_retain_kv_pages();
    const auto oom_table = limited.inspect_kv_page_table();
    const auto oom_state = limited.inspect_recurrent_state();
    bool exhausted = false;
    const std::array<std::uint32_t, 1> next{39};
    try {
      static_cast<void>(limited.run(next));
    } catch (const std::overflow_error &) {
      exhausted = true;
    }
    require(exhausted && limited.position() == 66 &&
                limited.inspect_kv_page_table() == oom_table &&
                limited.inspect_recurrent_state() == oom_state,
            "P05 physical OOM changed model frontier or state");
    limited.diagnostic_release_kv_pages(current_lease);
    limited.diagnostic_release_kv_pages(old_lease);
    const auto resumed = limited.run(next, {}, true);
    const auto expected = limited_teacher.run(next, {}, true);
    require(resumed.logits == expected.logits &&
                limited.inspect_recurrent_state() ==
                    limited_teacher.inspect_recurrent_state(),
            "P05 physical OOM recovery differs from teacher");
    std::cout << "P05 Qwen noncontiguous pages, abort and COW commit passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P05 Qwen paging failure: " << error.what() << '\n';
    return 1;
  }
}

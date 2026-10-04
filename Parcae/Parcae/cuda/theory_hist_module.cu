#include "theory_hist_module.hpp"

#include "parcae/core/sha256.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct TheoryHistModuleCache {
    std::mutex mu;
    std::unordered_map<std::string, TheoryHistModule::Entry> by_key;
    bool driver_ready = false;
    bool inject_oom = false;
    bool inject_invalid = false;
};

[[nodiscard]] TheoryHistModuleCache& module_cache() {
    static TheoryHistModuleCache cache;
    return cache;
}

[[nodiscard]] std::string module_cache_key(std::string_view uri, std::string_view digest) {
    std::string key;
    key.reserve(uri.size() + 1 + digest.size());
    key.append(uri);
    key.push_back('\n');
    key.append(digest);
    return key;
}

Status TheoryHistModule::driver_to_status(int result, std::string_view context) {
    const CUresult r = static_cast<CUresult>(result);
    if (r == CUDA_SUCCESS) {
        return Status::success();
    }
    const char* name = nullptr;
    const char* desc = nullptr;
    (void)cuGetErrorName(r, &name);
    (void)cuGetErrorString(r, &desc);
    std::string message;
    message.reserve(context.size() + 64);
    message.append(context);
    message.append(": ");
    message.append(name != nullptr ? name : "CUDA_ERROR");
    message.append(" — ");
    message.append(desc != nullptr ? desc : "unknown");
    return Status::error(std::move(message));
}

Status TheoryHistModule::nvrtc_to_status(int result, std::string_view context) {
    const nvrtcResult r = static_cast<nvrtcResult>(result);
    if (r == NVRTC_SUCCESS) {
        return Status::success();
    }
    const char* desc = nvrtcGetErrorString(r);
    std::string message;
    message.reserve(context.size() + 64);
    message.append(context);
    message.append(": ");
    message.append(desc != nullptr ? desc : "NVRTC_ERROR");
    return Status::error(std::move(message));
}

Status TheoryHistModule::init_driver() {
    TheoryHistModuleCache& cache = module_cache();
    if (cache.driver_ready) {
        return Status::success();
    }
    Status st = driver_to_status(cuInit(0), "TheoryHistModule::cuInit");
    if (!st.ok()) {
        return st;
    }
    // Retain primary context so cuModuleLoadData has a current context.
    CUdevice device = 0;
    st = driver_to_status(cuDeviceGet(&device, 0), "TheoryHistModule::cuDeviceGet");
    if (!st.ok()) {
        return st;
    }
    CUcontext ctx = nullptr;
    st = driver_to_status(cuDevicePrimaryCtxRetain(&ctx, device),
                          "TheoryHistModule::cuDevicePrimaryCtxRetain");
    if (!st.ok()) {
        return st;
    }
    st = driver_to_status(cuCtxSetCurrent(ctx), "TheoryHistModule::cuCtxSetCurrent");
    if (!st.ok()) {
        (void)cuDevicePrimaryCtxRelease(device);
        return st;
    }
    cache.driver_ready = true;
    return Status::success();
}

bool TheoryHistModule::has(std::string_view theory_uri) noexcept {
    TheoryHistModuleCache& cache = module_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    for (const auto& kv : cache.by_key) {
        if (kv.second.uri == theory_uri) {
            return true;
        }
    }
    return false;
}

bool TheoryHistModule::has(std::string_view theory_uri, std::string_view digest) noexcept {
    TheoryHistModuleCache& cache = module_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    return cache.by_key.find(module_cache_key(theory_uri, digest)) != cache.by_key.end();
}

std::size_t TheoryHistModule::cache_size() noexcept {
    TheoryHistModuleCache& cache = module_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    return cache.by_key.size();
}

void TheoryHistModule::clear() {
    TheoryHistModuleCache& cache = module_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    for (auto& kv : cache.by_key) {
        if (kv.second.module != nullptr) {
            (void)cuModuleUnload(static_cast<CUmodule>(kv.second.module));
            kv.second.module = nullptr;
            kv.second.function = nullptr;
        }
    }
    cache.by_key.clear();
    cache.inject_oom = false;
    cache.inject_invalid = false;
}

void TheoryHistModule::set_inject_oom(bool enabled) noexcept {
    module_cache().inject_oom = enabled;
}

void TheoryHistModule::set_inject_invalid_launch(bool enabled) noexcept {
    module_cache().inject_invalid = enabled;
}

Status TheoryHistModule::insert_loaded(std::string uri, std::string digest, LaunchKind kind,
                                       void* module, void* function, std::string symbol) {
    TheoryHistModuleCache& cache = module_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    const std::string key = module_cache_key(uri, digest);
    auto it = cache.by_key.find(key);
    if (it != cache.by_key.end()) {
        if (it->second.module != nullptr && it->second.module != module) {
            (void)cuModuleUnload(static_cast<CUmodule>(it->second.module));
        }
        it->second.kind = kind;
        it->second.module = module;
        it->second.function = function;
        it->second.symbol = std::move(symbol);
        return Status::success();
    }
    Entry entry;
    entry.uri = std::move(uri);
    entry.digest = std::move(digest);
    entry.kind = kind;
    entry.module = module;
    entry.function = function;
    entry.symbol = std::move(symbol);
    cache.by_key.emplace(key, std::move(entry));
    return Status::success();
}

TheoryHistModule::Entry* TheoryHistModule::find_ready(std::string_view theory_uri) noexcept {
    TheoryHistModuleCache& cache = module_cache();
    for (auto& kv : cache.by_key) {
        if (kv.second.uri == theory_uri) {
            return &kv.second;
        }
    }
    return nullptr;
}

Status TheoryHistModule::ensure_proxy(std::string_view theory_uri, std::string_view digest) {
    if (theory_uri.empty()) {
        return Status::error("TheoryHistModule::ensure_proxy: empty theory_uri");
    }
    if (digest.empty()) {
        return Status::error("TheoryHistModule::ensure_proxy: empty digest");
    }
    return insert_loaded(std::string(theory_uri), std::string(digest), LaunchKind::ProxyS0, nullptr,
                         nullptr, std::string{});
}

Status TheoryHistModule::load_module_image(std::string_view theory_uri, std::string_view digest,
                                           std::span<const std::uint8_t> image,
                                           std::string_view kernel_symbol, bool proxy_launch) {
    Status init = init_driver();
    if (!init.ok()) {
        return init;
    }

    CUmodule module = nullptr;
    Status load =
        driver_to_status(cuModuleLoadData(&module, image.data()), "TheoryHistModule::cuModuleLoadData");
    if (!load.ok()) {
        return load;
    }

    CUfunction fn = nullptr;
    LaunchKind kind = LaunchKind::ProxyS0;
    if (!proxy_launch) {
        if (kernel_symbol.empty()) {
            (void)cuModuleUnload(module);
            return Status::error("TheoryHistModule: empty kernel_symbol for driver launch");
        }
        Status gf = driver_to_status(cuModuleGetFunction(&fn, module, std::string(kernel_symbol).c_str()),
                                     "TheoryHistModule::cuModuleGetFunction");
        if (!gf.ok()) {
            (void)cuModuleUnload(module);
            return gf;
        }
        kind = LaunchKind::DriverKernel;
    }

    return insert_loaded(std::string(theory_uri), std::string(digest), kind, module, fn,
                         std::string(kernel_symbol));
}

Status TheoryHistModule::ensure_cubin(std::string_view theory_uri, std::string_view digest,
                                      std::span<const std::uint8_t> image,
                                      std::string_view kernel_symbol, bool proxy_launch) {
    if (theory_uri.empty()) {
        return Status::error("TheoryHistModule::ensure_cubin: empty theory_uri");
    }
    if (digest.size() != 64) {
        return Status::error("TheoryHistModule::ensure_cubin: digest must be 64 hex chars");
    }
    if (image.empty()) {
        return Status::error("TheoryHistModule::ensure_cubin: empty image");
    }
    const std::string got = Sha256::hex_digest(image.data(), image.size());
    if (got != digest) {
        return Status::error("TheoryHistModule::ensure_cubin: digest mismatch");
    }
    if (has(theory_uri, digest)) {
        return Status::success();
    }
    return load_module_image(theory_uri, digest, image, kernel_symbol, proxy_launch);
}

Status TheoryHistModule::ensure_nvrtc(std::string_view theory_uri, std::string_view digest,
                                      std::string_view cuda_source, std::string_view kernel_symbol,
                                      bool proxy_launch) {
    if (theory_uri.empty()) {
        return Status::error("TheoryHistModule::ensure_nvrtc: empty theory_uri");
    }
    if (digest.size() != 64) {
        return Status::error("TheoryHistModule::ensure_nvrtc: digest must be 64 hex chars");
    }
    if (cuda_source.empty()) {
        return Status::error("TheoryHistModule::ensure_nvrtc: empty cuda_source");
    }
    const std::string got = Sha256::hex_digest(cuda_source);
    if (got != digest) {
        return Status::error("TheoryHistModule::ensure_nvrtc: digest mismatch");
    }
    if (has(theory_uri, digest)) {
        return Status::success();
    }

    nvrtcProgram prog = nullptr;
    Status create = nvrtc_to_status(
        nvrtcCreateProgram(&prog, std::string(cuda_source).c_str(), "theory_hist_module.cu", 0,
                           nullptr, nullptr),
        "TheoryHistModule::nvrtcCreateProgram");
    if (!create.ok()) {
        return create;
    }

    // Prefer plate arch; fall back through common arches then no-arch.
    const char* arch_opts[] = {
        "--gpu-architecture=sm_120",
        "--gpu-architecture=compute_120",
        "--gpu-architecture=sm_90",
        "--gpu-architecture=sm_75",
    };
    nvrtcResult compile_rc = NVRTC_ERROR_COMPILATION;
    for (const char* arch : arch_opts) {
        const char* opts[] = {arch, "--std=c++17"};
        compile_rc = nvrtcCompileProgram(prog, 2, opts);
        if (compile_rc == NVRTC_SUCCESS) {
            break;
        }
        // Program is spent after a failed compile — recreate.
        (void)nvrtcDestroyProgram(&prog);
        create = nvrtc_to_status(
            nvrtcCreateProgram(&prog, std::string(cuda_source).c_str(), "theory_hist_module.cu", 0,
                               nullptr, nullptr),
            "TheoryHistModule::nvrtcCreateProgram(retry)");
        if (!create.ok()) {
            return create;
        }
    }
    if (compile_rc != NVRTC_SUCCESS) {
        compile_rc = nvrtcCompileProgram(prog, 0, nullptr);
    }
    if (compile_rc != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        (void)nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) {
            (void)nvrtcGetProgramLog(prog, log.data());
        }
        (void)nvrtcDestroyProgram(&prog);
        return Status::error(std::string("TheoryHistModule::nvrtcCompileProgram failed: ") + log);
    }

    std::size_t ptx_size = 0;
    Status psz =
        nvrtc_to_status(nvrtcGetPTXSize(prog, &ptx_size), "TheoryHistModule::nvrtcGetPTXSize");
    if (!psz.ok()) {
        (void)nvrtcDestroyProgram(&prog);
        return psz;
    }
    std::vector<char> ptx(ptx_size);
    Status ptx_st =
        nvrtc_to_status(nvrtcGetPTX(prog, ptx.data()), "TheoryHistModule::nvrtcGetPTX");
    (void)nvrtcDestroyProgram(&prog);
    if (!ptx_st.ok()) {
        return ptx_st;
    }

    std::vector<std::uint8_t> image(ptx.begin(), ptx.end());
    // Cache key stays source digest; PTX image is not re-hashed.
    return load_module_image(theory_uri, digest, image, kernel_symbol, proxy_launch);
}

StatusOr<std::string> TheoryHistModule::read_file_bytes(const std::filesystem::path& path,
                                                        std::vector<std::uint8_t>* out) {
    if (out == nullptr) {
        return Status::error("TheoryHistModule::read_file_bytes: null out");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status::error("TheoryHistModule::read_file_bytes: open failed: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff len = in.tellg();
    if (len < 0) {
        return Status::error("TheoryHistModule::read_file_bytes: size failed: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    out->assign(static_cast<std::size_t>(len), 0);
    if (len > 0) {
        in.read(reinterpret_cast<char*>(out->data()), len);
        if (!in) {
            return Status::error("TheoryHistModule::read_file_bytes: read failed: " + path.string());
        }
    }
    return Sha256::hex_digest(out->data(), out->size());
}

Status TheoryHistModule::ensure_file(std::string_view theory_uri, const std::filesystem::path& path,
                                     std::string_view kernel_symbol, bool proxy_launch) {
    std::vector<std::uint8_t> bytes;
    StatusOr<std::string> digest = read_file_bytes(path, &bytes);
    if (!digest.ok()) {
        return digest.status();
    }
    return ensure_cubin(theory_uri, digest.value(), bytes, kernel_symbol, proxy_launch);
}

Status TheoryHistModule::launch_async(
    std::string_view theory_uri, const std::uint8_t* device_in, const std::uint8_t* device_ops,
    const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
    std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
    std::uint8_t binds_index_i, std::uint16_t max_stack, const double* device_probabilities,
    std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
    std::size_t candidate_count, std::size_t token_count, cudaStream_t stream) {
    TheoryHistModuleCache& cache = module_cache();
    if (cache.inject_oom) {
        return Status::error(
            "TheoryHistModule::launch_async: cudaErrorMemoryAllocation — injected OOM");
    }
    if (cache.inject_invalid) {
        return Status::error("TheoryHistModule::launch_async: invalid module — injected failure");
    }

    LaunchKind kind = LaunchKind::ProxyS0;
    CUfunction fn = nullptr;
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        Entry* entry = find_ready(theory_uri);
        if (entry == nullptr) {
            return Status::error("TheoryHistModule::launch_async: no module for URI");
        }
        kind = entry->kind;
        fn = static_cast<CUfunction>(entry->function);
    }

    if (kind == LaunchKind::ProxyS0) {
        return TheoryChi2Batch::launch_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, binds_index_i, max_stack, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count, stream);
    }

    if (fn == nullptr) {
        return Status::error("TheoryHistModule::launch_async: null kernel function");
    }

    const std::uint8_t* in_arg = device_in;
    const std::uint8_t* ops_arg = device_ops;
    const std::uint8_t* imm_arg = device_imm;
    const std::uint8_t* slots_arg = device_slots;
    const double* probs_arg = device_probabilities;
    std::uint32_t* counts_arg = device_counts;
    double* scores_arg = device_scores;
    std::uint8_t* err_arg = device_lane_err;
    std::uint32_t op_count_arg = op_count;
    std::uint16_t slot_count_arg = slot_count;
    std::uint16_t cipher_slot_arg = cipher_slot;
    std::uint16_t index_slot_arg = index_slot;
    std::uint8_t binds_arg = binds_index_i;
    std::uint16_t max_stack_arg = max_stack;
    std::size_t cand_arg = candidate_count;
    std::size_t tok_arg = token_count;

    void* args[] = {&in_arg,         &ops_arg,          &imm_arg,         &op_count_arg,
                    &slots_arg,      &slot_count_arg,   &cipher_slot_arg, &index_slot_arg,
                    &binds_arg,      &max_stack_arg,    &probs_arg,       &counts_arg,
                    &scores_arg,     &err_arg,          &cand_arg,        &tok_arg};

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    return driver_to_status(
        cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, cu_stream, args, nullptr),
        "TheoryHistModule::cuLaunchKernel");
}

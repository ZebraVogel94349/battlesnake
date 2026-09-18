#include "../header/battlesnake_cuda.h"

#include <string>

static std::string g_hisss_cuda_error =
    "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";

bool hisss_cuda_available() {
    return false;
}

const char* hisss_cuda_last_error() {
    return g_hisss_cuda_error.c_str();
}

int hisss_cuda_run_random_rollouts(
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    const int*,
    int,
    const int*,
    int,
    const int*,
    const int*,
    const bool*,
    const int*,
    const int*,
    const int*,
    bool,
    bool,
    int,
    int,
    const bool*,
    int,
    int,
    unsigned long long,
    int*,
    bool*,
    int*,
    bool*,
    int*,
    int*,
    int*,
    int*,
    int*
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

void* hisss_cuda_blackout_vec_create(int, unsigned long long, int) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return nullptr;
}

void hisss_cuda_blackout_vec_close(void*) {}

int hisss_cuda_blackout_vec_reset(void*, float*, bool*) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_step(void*, const int*, float*, float*, bool*, bool*) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_reset_device(void*, uintptr_t, uintptr_t) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_reset_all_device(void*, uintptr_t, uintptr_t) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_reset_all_device_async(
    void*, uintptr_t, uintptr_t, bool, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_step_device(
    void*,
    uintptr_t,
    uintptr_t,
    uintptr_t,
    uintptr_t,
    uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_step_all_device(
    void*,
    uintptr_t,
    uintptr_t,
    uintptr_t,
    uintptr_t,
    uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_step_all_device_async(
    void*, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, bool, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_step_all_eval_device(
    void*, uintptr_t, int, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    uintptr_t, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_best_actions_device(void*, uintptr_t, uintptr_t) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_best_actions_device_async(
    void*, uintptr_t, uintptr_t, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_heuristic_actions_device(
    void*, uintptr_t, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

int hisss_cuda_blackout_vec_heuristic_actions_device_async(
    void*, uintptr_t, uintptr_t, uintptr_t
) {
    g_hisss_cuda_error =
        "hisss was built without CUDA support. Install the CUDA toolkit and rebuild.";
    return 1;
}

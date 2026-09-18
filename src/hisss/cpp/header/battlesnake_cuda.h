#ifndef HISSS_BATTLESNAKE_CUDA_H
#define HISSS_BATTLESNAKE_CUDA_H

#include <stdbool.h>
#include <stdint.h>

bool hisss_cuda_available();
const char* hisss_cuda_last_error();

int hisss_cuda_run_random_rollouts(
    int w,
    int h,
    int num_snakes,
    int min_food,
    int food_spawn_chance,
    int init_turns_played,
    bool spawn_snakes_randomly,
    const int* snake_body_lengths,
    int max_init_body_length,
    const int* snake_bodies,
    int num_init_food,
    const int* food_spawns,
    const int* food_spawn_turn_values,
    const bool* snake_alive,
    const int* snake_health,
    const int* snake_len,
    const int* max_health,
    bool wrapped,
    bool royale,
    int shrink_n_turns,
    int hazard_damage,
    const bool* init_hazards,
    int num_games,
    int max_turns,
    unsigned long long seed,
    int* out_turns_played,
    bool* out_terminal,
    int* out_winner,
    bool* out_alive,
    int* out_lengths,
    int* out_health,
    int* out_death_cause,
    int* out_death_turn,
    int* out_killer_id
);

void* hisss_cuda_blackout_vec_create(
    int num_envs,
    unsigned long long seed,
    int duel_probability_ppm
);
void hisss_cuda_blackout_vec_close(void* env);
int hisss_cuda_blackout_vec_reset(
    void* env,
    float* out_obs,
    bool* out_legal_mask
);
int hisss_cuda_blackout_vec_step(
    void* env,
    const int* actions,
    float* out_obs,
    float* out_rewards,
    bool* out_done,
    bool* out_legal_mask
);
int hisss_cuda_blackout_vec_reset_device(
    void* env,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr
);
int hisss_cuda_blackout_vec_reset_all_device(
    void* env,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr
);
int hisss_cuda_blackout_vec_reset_all_device_async(
    void* env,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    bool out_obs_fp16,
    uintptr_t stream_ptr
);
int hisss_cuda_blackout_vec_step_device(
    void* env,
    uintptr_t actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr
);
int hisss_cuda_blackout_vec_step_all_device(
    void* env,
    uintptr_t all_actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr
);
int hisss_cuda_blackout_vec_step_all_device_async(
    void* env,
    uintptr_t all_actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    bool out_obs_fp16,
    uintptr_t stream_ptr
);
int hisss_cuda_blackout_vec_step_all_eval_device(
    void* env,
    uintptr_t all_actions_device_ptr,
    int max_turns,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    uintptr_t out_winner_device_ptr,
    uintptr_t out_alive_device_ptr,
    uintptr_t out_turns_device_ptr
);
int hisss_cuda_blackout_vec_best_actions_device(
    void* env,
    uintptr_t selection_mask_device_ptr,
    uintptr_t out_actions_device_ptr
);
int hisss_cuda_blackout_vec_best_actions_device_async(
    void* env,
    uintptr_t selection_mask_device_ptr,
    uintptr_t out_actions_device_ptr,
    uintptr_t stream_ptr
);
int hisss_cuda_blackout_vec_heuristic_actions_device(
    void* env,
    uintptr_t profile_ids_device_ptr,
    uintptr_t out_actions_device_ptr
);
int hisss_cuda_blackout_vec_heuristic_actions_device_async(
    void* env,
    uintptr_t profile_ids_device_ptr,
    uintptr_t out_actions_device_ptr,
    uintptr_t stream_ptr
);

#endif

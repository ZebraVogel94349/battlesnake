//
// Created by mahla on 24/10/2022.
//

#include <iostream>
#include "../header/battlesnake.h"
#include "../header/battlesnake_helper.h"
#include "../header/battlesnake_cuda.h"
#include "../header/nash.h"

#if defined(_WIN32) || defined(_MSC_VER)
    #define HISSS_EXPORT __declspec(dllexport)
#else
    #define HISSS_EXPORT __attribute__((visibility("default")))
#endif

//g++ -c -fPIC link.cpp -o link.o
//g++ -shared -Wl,-soname,-liblink.so -o liblink.so link.o

extern "C" {
    HISSS_EXPORT GameState* init_cpp(
        int w,
        int h,
        int num_snakes,
        int min_food,
        int food_spawn_chance,
        int init_turns_played,
        bool spawn_snakes_randomly,
        int* snake_body_lengths,
        int max_body_length,
        int* snake_bodies,
        int num_init_food,
        int* food_spawns,
        int* food_spawn_turn_values,
        bool* snake_alive,
        int* snake_health,
        int* snake_len,
        int* max_health,
        bool wrapped,
        bool royale,
        int shrink_n_turns,
        int hazard_damage,
        bool* init_hazards
    ){
        return init(
            w,
            h,
            num_snakes,
            min_food,
            food_spawn_chance,
            init_turns_played,
            spawn_snakes_randomly,
            snake_body_lengths,
            max_body_length,
            snake_bodies,
            num_init_food,
            food_spawns,
            food_spawn_turn_values,
            snake_alive,
            snake_health,
            snake_len,
            max_health,
            wrapped,
            royale,
            shrink_n_turns,
            hazard_damage,
            init_hazards
        );
    }
    HISSS_EXPORT void step_cpp(GameState* state, int* actions){
        step(state, actions);
    }
    HISSS_EXPORT void str_cpp(GameState* state, char* arr){
        draw_to_arr(state, arr);
    }
    HISSS_EXPORT void custom_encode_cpp(
            GameState* state,
            float* arr,
            bool include_current_food,
            bool include_next_food,
            bool include_board,
            bool include_number_of_turns,
            bool flatten_snakes,
            int player_snake,
            bool include_snake_body_as_one_hot,
            bool include_snake_body,
            bool include_snake_head,
            bool include_snake_tail,
            bool include_snake_health,
            bool include_snake_length,
            bool centered,
            bool include_distance_map,
            bool include_area_control,
            bool include_food_distance,
            bool include_hazards,
            bool include_tail_distance,
            bool include_num_food_on_board,
            float fixed_food_spawn_chance,
            bool include_temperatures,
            bool single_temperature,
            const float* temperatures
    ){
        construct_custom_encoding(
                state,
                arr,
                include_current_food,
                include_next_food,
                include_board,
                include_number_of_turns,
                flatten_snakes,
                player_snake,
                include_snake_body_as_one_hot,
                include_snake_body,
                include_snake_head,
                include_snake_tail,
                include_snake_health,
                include_snake_length,
                centered,
                include_distance_map,
                include_area_control,
                include_food_distance,
                include_hazards,
                include_tail_distance,
                include_num_food_on_board,
                fixed_food_spawn_chance,
                include_temperatures,
                single_temperature,
                temperatures
            );
    }

    HISSS_EXPORT GameState* clone_cpp(GameState* state){
        return clone(state);
    }
    HISSS_EXPORT void close_cpp(GameState* state){
        close(state);
    }
    HISSS_EXPORT void actions_cpp(GameState* state, int snake_id, int* actions){
        legal_actions(state, snake_id, actions);
    }
    HISSS_EXPORT bool equals_cpp(GameState* state1, GameState* state2){
        return equals(state1, state2);
    }
    HISSS_EXPORT void alive_cpp(GameState* state, bool* arr){
        alive(state, arr);
    }
    HISSS_EXPORT void snake_length_cpp(GameState* state, int* arr){
        snake_length(state, arr);
    }
    HISSS_EXPORT int snake_body_length_cpp(GameState* state, int player){
        return snake_body_length(state, player);
    }
    HISSS_EXPORT void snake_pos_cpp(GameState* state, int player, int* arr){
        snake_pos(state, player, arr);
    }
    HISSS_EXPORT void snake_health_cpp(GameState* state, int* arr){
        snake_health(state, arr);
    }
    HISSS_EXPORT int num_food_cpp(GameState* state){
        return num_food(state);
    }
    HISSS_EXPORT void food_pos_cpp(GameState* state, int* arr){
        food_pos(state, arr);
    }
    HISSS_EXPORT void food_spawn_turns_cpp(GameState* state, int* arr){
        food_spawn_turns_fn(state, arr);
    }
    HISSS_EXPORT int turns_played_cpp(GameState* state){
        return turns_played(state);
    }
    HISSS_EXPORT void area_control_cpp(
            GameState* state,
            float* area_arr,
            int* food_dist_arr,
            int* tail_dist_arr,
            bool* reached_tail,
            bool* reached_food,
            float weight,
            float food_weight,
            float hazard_weight,
            float food_in_hazard_weight
    ){
        area_control(
                state,
                area_arr,
                food_dist_arr,
                tail_dist_arr,
                reached_tail,
                reached_food,
                weight,
                food_weight,
                hazard_weight,
                food_in_hazard_weight
        );
    }
    HISSS_EXPORT void hazards_cpp(GameState* state, bool* arr){
        hazards(state, arr);
    }

    HISSS_EXPORT int compute_nash_cpp(
            int num_player_at_turn,
            const int* num_available_actions,  //shape (num_player_at_turn,)
            const int* available_actions,  // shape (sum(num_available_actions))
            const int* joint_actions, // shape (prod(num_available_actions) * num_player)
            const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
            double* result_values,  // shape (num_players)
            double* result_policies // shape (sum(num_available_actions))
    ){
        if (num_player_at_turn == 2){
            int result = compute_2p_nash(
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                result_values,
                result_policies
            );
            return result;
        } else {
            int result = compute_nash(
                num_player_at_turn,
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                result_values,
                result_policies
            );
            return result;
        }
    }

    HISSS_EXPORT int snake_elim_cause_cpp(GameState* state, int snake_id) {
        return state->snakes.at(snake_id)->death_cause;
    }
    HISSS_EXPORT int snake_elim_killer_cpp(GameState* state, int snake_id) {
        return state->snakes.at(snake_id)->killer_id;
    }
    HISSS_EXPORT int snake_elim_turn_cpp(GameState* state, int snake_id) {
        return state->snakes.at(snake_id)->death_turn;
    }
    HISSS_EXPORT void set_elim_info_cpp(GameState* state, int snake_id, int cause, int killer_id, int death_turn) {
        Snake* s = state->snakes.at(snake_id);
        s->death_cause = cause;
        s->killer_id = killer_id;
        s->death_turn = death_turn;
    }

    HISSS_EXPORT void set_seed(int seed) {
        set_seed_gym(seed);
        set_seed_utils(seed);
    }

    HISSS_EXPORT void char_game_matrix_cpp(
            GameState* state,
            char* matrix
    ){
       char_game_matrix(state, matrix);
    }

    HISSS_EXPORT bool cuda_available_cpp() {
        return hisss_cuda_available();
    }

    HISSS_EXPORT const char* cuda_last_error_cpp() {
        return hisss_cuda_last_error();
    }

    HISSS_EXPORT int cuda_run_random_rollouts_cpp(
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
    ) {
        return hisss_cuda_run_random_rollouts(
            w,
            h,
            num_snakes,
            min_food,
            food_spawn_chance,
            init_turns_played,
            spawn_snakes_randomly,
            snake_body_lengths,
            max_init_body_length,
            snake_bodies,
            num_init_food,
            food_spawns,
            food_spawn_turn_values,
            snake_alive,
            snake_health,
            snake_len,
            max_health,
            wrapped,
            royale,
            shrink_n_turns,
            hazard_damage,
            init_hazards,
            num_games,
            max_turns,
            seed,
            out_turns_played,
            out_terminal,
            out_winner,
            out_alive,
            out_lengths,
            out_health,
            out_death_cause,
            out_death_turn,
            out_killer_id
        );
    }

    HISSS_EXPORT void* cuda_blackout_vec_create_cpp(
        int num_envs,
        unsigned long long seed,
        int duel_probability_ppm
    ) {
        return hisss_cuda_blackout_vec_create(
            num_envs, seed, duel_probability_ppm
        );
    }

    HISSS_EXPORT void cuda_blackout_vec_close_cpp(void* env) {
        hisss_cuda_blackout_vec_close(env);
    }

    HISSS_EXPORT int cuda_blackout_vec_reset_cpp(
        void* env,
        float* out_obs,
        bool* out_legal_mask
    ) {
        return hisss_cuda_blackout_vec_reset(env, out_obs, out_legal_mask);
    }

    HISSS_EXPORT int cuda_blackout_vec_step_cpp(
        void* env,
        const int* actions,
        float* out_obs,
        float* out_rewards,
        bool* out_done,
        bool* out_legal_mask
    ) {
        return hisss_cuda_blackout_vec_step(
            env,
            actions,
            out_obs,
            out_rewards,
            out_done,
            out_legal_mask
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_reset_device_cpp(
        void* env,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_legal_mask_device_ptr
    ) {
        return hisss_cuda_blackout_vec_reset_device(
            env,
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_reset_all_device_cpp(
        void* env,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_legal_mask_device_ptr
    ) {
        return hisss_cuda_blackout_vec_reset_all_device(
            env,
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_reset_all_device_async_cpp(
        void* env,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_legal_mask_device_ptr,
        bool out_obs_fp16,
        unsigned long long stream_ptr
    ) {
        return hisss_cuda_blackout_vec_reset_all_device_async(
            env,
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr),
            out_obs_fp16,
            static_cast<uintptr_t>(stream_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_step_device_cpp(
        void* env,
        unsigned long long actions_device_ptr,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_rewards_device_ptr,
        unsigned long long out_done_device_ptr,
        unsigned long long out_legal_mask_device_ptr
    ) {
        return hisss_cuda_blackout_vec_step_device(
            env,
            static_cast<uintptr_t>(actions_device_ptr),
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_rewards_device_ptr),
            static_cast<uintptr_t>(out_done_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_step_all_device_cpp(
        void* env,
        unsigned long long all_actions_device_ptr,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_rewards_device_ptr,
        unsigned long long out_done_device_ptr,
        unsigned long long out_legal_mask_device_ptr
    ) {
        return hisss_cuda_blackout_vec_step_all_device(
            env,
            static_cast<uintptr_t>(all_actions_device_ptr),
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_rewards_device_ptr),
            static_cast<uintptr_t>(out_done_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_step_all_device_async_cpp(
        void* env,
        unsigned long long all_actions_device_ptr,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_rewards_device_ptr,
        unsigned long long out_done_device_ptr,
        unsigned long long out_legal_mask_device_ptr,
        bool out_obs_fp16,
        unsigned long long stream_ptr
    ) {
        return hisss_cuda_blackout_vec_step_all_device_async(
            env,
            static_cast<uintptr_t>(all_actions_device_ptr),
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_rewards_device_ptr),
            static_cast<uintptr_t>(out_done_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr),
            out_obs_fp16,
            static_cast<uintptr_t>(stream_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_step_all_eval_device_cpp(
        void* env,
        unsigned long long all_actions_device_ptr,
        int max_turns,
        unsigned long long out_obs_device_ptr,
        unsigned long long out_done_device_ptr,
        unsigned long long out_legal_mask_device_ptr,
        unsigned long long out_winner_device_ptr,
        unsigned long long out_alive_device_ptr,
        unsigned long long out_turns_device_ptr
    ) {
        return hisss_cuda_blackout_vec_step_all_eval_device(
            env,
            static_cast<uintptr_t>(all_actions_device_ptr),
            max_turns,
            static_cast<uintptr_t>(out_obs_device_ptr),
            static_cast<uintptr_t>(out_done_device_ptr),
            static_cast<uintptr_t>(out_legal_mask_device_ptr),
            static_cast<uintptr_t>(out_winner_device_ptr),
            static_cast<uintptr_t>(out_alive_device_ptr),
            static_cast<uintptr_t>(out_turns_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_best_actions_device_cpp(
        void* env,
        unsigned long long selection_mask_device_ptr,
        unsigned long long out_actions_device_ptr
    ) {
        return hisss_cuda_blackout_vec_best_actions_device(
            env,
            static_cast<uintptr_t>(selection_mask_device_ptr),
            static_cast<uintptr_t>(out_actions_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_best_actions_device_async_cpp(
        void* env,
        unsigned long long selection_mask_device_ptr,
        unsigned long long out_actions_device_ptr,
        unsigned long long stream_ptr
    ) {
        return hisss_cuda_blackout_vec_best_actions_device_async(
            env,
            static_cast<uintptr_t>(selection_mask_device_ptr),
            static_cast<uintptr_t>(out_actions_device_ptr),
            static_cast<uintptr_t>(stream_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_heuristic_actions_device_cpp(
        void* env,
        unsigned long long profile_ids_device_ptr,
        unsigned long long out_actions_device_ptr
    ) {
        return hisss_cuda_blackout_vec_heuristic_actions_device(
            env,
            static_cast<uintptr_t>(profile_ids_device_ptr),
            static_cast<uintptr_t>(out_actions_device_ptr)
        );
    }

    HISSS_EXPORT int cuda_blackout_vec_heuristic_actions_device_async_cpp(
        void* env,
        unsigned long long profile_ids_device_ptr,
        unsigned long long out_actions_device_ptr,
        unsigned long long stream_ptr
    ) {
        return hisss_cuda_blackout_vec_heuristic_actions_device_async(
            env,
            static_cast<uintptr_t>(profile_ids_device_ptr),
            static_cast<uintptr_t>(out_actions_device_ptr),
            static_cast<uintptr_t>(stream_ptr)
        );
    }

}

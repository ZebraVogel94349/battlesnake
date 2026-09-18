#include "../header/battlesnake_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

namespace {

constexpr int HISSS_CUDA_MAX_SNAKES = 16;
// The simulator assigns one independent game (or snake seat) to each thread.
// A single-warp block exposes many more blocks to wide GPUs than the previous
// 128-thread launch without introducing partially occupied warps.
constexpr int HISSS_CUDA_GAME_BLOCK_SIZE = 32;
constexpr int HISSS_CUDA_DENSE_BLOCK_SIZE = 256;

std::string g_hisss_cuda_error;

struct SimParams {
    int w;
    int h;
    int cells;
    int num_snakes;
    int min_food;
    int food_spawn_chance;
    int init_turns_played;
    bool spawn_snakes_randomly;
    int max_init_body_length;
    int num_init_food;
    bool wrapped;
    bool royale;
    int shrink_n_turns;
    int hazard_damage;
    int num_games;
    int max_turns;
    int max_body_len;
    int max_food;
    int duel_probability_ppm;
};

struct InitArrays {
    int* snake_body_lengths;
    int* snake_bodies;
    int* food_spawns;
    int* food_spawn_turn_values;
    bool* snake_alive;
    int* snake_health;
    int* snake_len;
    int* max_health;
    bool* init_hazards;
};

struct DeviceArrays {
    int* body_x;
    int* body_y;
    int* body_size;
    unsigned char* board;
    int* food_x;
    int* food_y;
    int* food_turn;
    int* food_count;
    unsigned char* scratch;
    unsigned char* hazards;
    unsigned char* alive;
    int* health;
    int* length;
    int* max_health;
    int* death_cause;
    int* death_turn;
    int* killer_id;
    int* out_turns_played;
    unsigned char* out_terminal;
    int* out_winner;
};

void set_last_error(const std::string& msg) {
    g_hisss_cuda_error = msg;
}

void set_cuda_error(const char* op, cudaError_t err) {
    std::ostringstream oss;
    oss << op << " failed: " << cudaGetErrorString(err);
    set_last_error(oss.str());
}

template <typename T>
bool device_alloc(T** ptr, size_t count, const char* name) {
    *ptr = nullptr;
    if (count == 0) {
        return true;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(" << name << ") failed: " << cudaGetErrorString(err);
        set_last_error(oss.str());
        return false;
    }
    return true;
}

template <typename T>
bool device_alloc_copy(T** ptr, const T* src, size_t count, const char* name) {
    *ptr = nullptr;
    if (count == 0) {
        return true;
    }
    if (src == nullptr) {
        std::ostringstream oss;
        oss << "missing required input array: " << name;
        set_last_error(oss.str());
        return false;
    }
    if (!device_alloc(ptr, count, name)) {
        return false;
    }
    cudaError_t err = cudaMemcpy(*ptr, src, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemcpy(" << name << ") failed: " << cudaGetErrorString(err);
        set_last_error(oss.str());
        cudaFree(*ptr);
        *ptr = nullptr;
        return false;
    }
    return true;
}

void free_init_arrays(InitArrays& init) {
    cudaFree(init.snake_body_lengths);
    cudaFree(init.snake_bodies);
    cudaFree(init.food_spawns);
    cudaFree(init.food_spawn_turn_values);
    cudaFree(init.snake_alive);
    cudaFree(init.snake_health);
    cudaFree(init.snake_len);
    cudaFree(init.max_health);
    cudaFree(init.init_hazards);
}

void free_device_arrays(DeviceArrays& a) {
    cudaFree(a.body_x);
    cudaFree(a.body_y);
    cudaFree(a.body_size);
    cudaFree(a.board);
    cudaFree(a.food_x);
    cudaFree(a.food_y);
    cudaFree(a.food_turn);
    cudaFree(a.food_count);
    cudaFree(a.scratch);
    cudaFree(a.hazards);
    cudaFree(a.alive);
    cudaFree(a.health);
    cudaFree(a.length);
    cudaFree(a.max_health);
    cudaFree(a.death_cause);
    cudaFree(a.death_turn);
    cudaFree(a.killer_id);
    cudaFree(a.out_turns_played);
    cudaFree(a.out_terminal);
    cudaFree(a.out_winner);
}

__device__ unsigned long long splitmix64_next(unsigned long long* state) {
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ unsigned int random_u32(unsigned long long* state) {
    return static_cast<unsigned int>(splitmix64_next(state) >> 32);
}

__device__ int random_mod(unsigned long long* state, int n) {
    if (n <= 1) {
        return 0;
    }
    return static_cast<int>(random_u32(state) % static_cast<unsigned int>(n));
}

__device__ int snake_index(const SimParams& p, int game, int snake) {
    return game * p.num_snakes + snake;
}

__device__ int body_index(const SimParams& p, int game, int snake, int body_pos) {
    return (game * p.num_snakes + snake) * p.max_body_len + body_pos;
}

__device__ int board_index(const SimParams& p, int game, int snake, int cell) {
    return (game * p.num_snakes + snake) * p.cells + cell;
}

__device__ int game_cell_index(const SimParams& p, int game, int cell) {
    return game * p.cells + cell;
}

__device__ int food_index(const SimParams& p, int game, int food) {
    return game * p.max_food + food;
}

__device__ int cell_index(const SimParams& p, int x, int y) {
    return y * p.w + x;
}

__device__ bool in_bounds(const SimParams& p, int x, int y) {
    return x >= 0 && y >= 0 && x < p.w && y < p.h;
}

__device__ void new_position(
    const SimParams& p,
    int x,
    int y,
    int move,
    int* nx,
    int* ny
) {
    *nx = x;
    *ny = y;
    if (move == 1) {
        *nx = x + 1;
        if (p.wrapped && *nx == p.w) {
            *nx = 0;
        }
    } else if (move == 2) {
        *ny = y - 1;
        if (p.wrapped && *ny == -1) {
            *ny = p.h - 1;
        }
    } else if (move == 3) {
        *nx = x - 1;
        if (p.wrapped && *nx == -1) {
            *nx = p.w - 1;
        }
    } else {
        *ny = y + 1;
        if (p.wrapped && *ny == p.h) {
            *ny = 0;
        }
    }
}

__device__ bool food_exists(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int x,
    int y
) {
    int count = a.food_count[game];
    for (int i = 0; i < count; ++i) {
        int idx = food_index(p, game, i);
        if (a.food_x[idx] == x && a.food_y[idx] == y) {
            return true;
        }
    }
    return false;
}

__device__ void shuffle4(int* xs, int* ys, unsigned long long* rng_state) {
    for (int i = 3; i > 0; --i) {
        int j = random_mod(rng_state, i + 1);
        int tx = xs[i];
        int ty = ys[i];
        xs[i] = xs[j];
        ys[i] = ys[j];
        xs[j] = tx;
        ys[j] = ty;
    }
}

__device__ void initialize_random_snakes(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    unsigned long long* rng_state
) {
    int mx = p.w - 2;
    int md = (p.w - 1) / 2;
    int corner_x[4];
    int corner_y[4];
    int side_x[4];
    int side_y[4];

    if (p.w == 3) {
        corner_x[0] = 0; corner_y[0] = 0;
        corner_x[1] = 0; corner_y[1] = 2;
        corner_x[2] = 2; corner_y[2] = 0;
        corner_x[3] = 2; corner_y[3] = 2;
        side_x[0] = 0; side_y[0] = 1;
        side_x[1] = 1; side_y[1] = 0;
        side_x[2] = 2; side_y[2] = 1;
        side_x[3] = 1; side_y[3] = 2;
    } else {
        corner_x[0] = 1;  corner_y[0] = 1;
        corner_x[1] = 1;  corner_y[1] = mx;
        corner_x[2] = mx; corner_y[2] = 1;
        corner_x[3] = mx; corner_y[3] = mx;
        side_x[0] = 1;  side_y[0] = md;
        side_x[1] = md; side_y[1] = 1;
        side_x[2] = md; side_y[2] = mx;
        side_x[3] = mx; side_y[3] = md;
    }

    shuffle4(corner_x, corner_y, rng_state);
    shuffle4(side_x, side_y, rng_state);
    bool corner_first = random_mod(rng_state, 2) != 0;
    int counter = 0;

    for (int phase = 0; phase < 2; ++phase) {
        bool use_corner = corner_first ? (phase == 0) : (phase == 1);
        for (int i = 0; i < 4 && counter < p.num_snakes; ++i) {
            int x = use_corner ? corner_x[i] : side_x[i];
            int y = use_corner ? corner_y[i] : side_y[i];
            int si = snake_index(p, game, counter);
            int bi = body_index(p, game, counter, 0);
            a.body_x[bi] = x;
            a.body_y[bi] = y;
            a.body_size[si] = 1;
            if (in_bounds(p, x, y)) {
                a.board[board_index(p, game, counter, cell_index(p, x, y))] = 1;
            }
            counter++;
        }
    }
}

__device__ bool is_away_from_center(int fx, int fy, int sx, int sy, int cx, int cy) {
    return (fx < sx && sx < cx) || (fx > sx && sx > cx) ||
           (fy < sy && sy < cy) || (fy > sy && sy > cy);
}

__device__ void initialize_random_food(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    unsigned long long* rng_state
) {
    int mid_x = (p.w - 1) / 2;
    int mid_y = (p.h - 1) / 2;
    int idx = food_index(p, game, 0);
    a.food_x[idx] = mid_x;
    a.food_y[idx] = mid_y;
    a.food_turn[idx] = turn;
    a.food_count[game] = 1;

    if (p.num_snakes > 4 || p.w == 3) {
        return;
    }

    for (int s = 0; s < p.num_snakes && a.food_count[game] < p.max_food; ++s) {
        if (!a.alive[snake_index(p, game, s)]) {
            continue;
        }
        int head_idx = body_index(p, game, s, 0);
        int sx = a.body_x[head_idx];
        int sy = a.body_y[head_idx];
        int cand_x[4] = {sx - 1, sx - 1, sx + 1, sx + 1};
        int cand_y[4] = {sy - 1, sy + 1, sy - 1, sy + 1};
        int valid_x[4];
        int valid_y[4];
        int valid_count = 0;

        for (int i = 0; i < 4; ++i) {
            int fx = cand_x[i];
            int fy = cand_y[i];
            if (fx == mid_x && fy == mid_y) {
                continue;
            }
            if (food_exists(p, a, game, fx, fy)) {
                continue;
            }
            if (!is_away_from_center(fx, fy, sx, sy, mid_x, mid_y)) {
                continue;
            }
            if ((fx == 0 || fx == p.w - 1) && (fy == 0 || fy == p.h - 1)) {
                continue;
            }
            valid_x[valid_count] = fx;
            valid_y[valid_count] = fy;
            valid_count++;
        }

        if (valid_count > 0) {
            int chosen = random_mod(rng_state, valid_count);
            int food_count = a.food_count[game];
            int fidx = food_index(p, game, food_count);
            a.food_x[fidx] = valid_x[chosen];
            a.food_y[fidx] = valid_y[chosen];
            a.food_turn[fidx] = turn;
            a.food_count[game] = food_count + 1;
        }
    }
}

__device__ void initialize_game(
    const SimParams& p,
    const InitArrays& init,
    const DeviceArrays& a,
    int game,
    unsigned long long* rng_state
) {
    bool duel_game = p.duel_probability_ppm > 0 &&
        random_mod(rng_state, 1000000) < p.duel_probability_ppm;
    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        a.alive[si] = init.snake_alive[s] && (!duel_game || s < 2) ? 1 : 0;
        a.health[si] = init.snake_health[s];
        a.length[si] = init.snake_len[s];
        a.max_health[si] = init.max_health[s];
        a.death_cause[si] = 0;
        a.death_turn[si] = -1;
        a.killer_id[si] = -1;
        a.body_size[si] = 0;
        for (int i = 0; i < p.max_body_len; ++i) {
            int bi = body_index(p, game, s, i);
            a.body_x[bi] = -1;
            a.body_y[bi] = -1;
        }
        for (int cell = 0; cell < p.cells; ++cell) {
            a.board[board_index(p, game, s, cell)] = 0;
        }
    }

    for (int cell = 0; cell < p.cells; ++cell) {
        a.hazards[game_cell_index(p, game, cell)] =
            (init.init_hazards != nullptr && init.init_hazards[cell]) ? 1 : 0;
        a.scratch[game_cell_index(p, game, cell)] = 0;
    }

    if (p.spawn_snakes_randomly) {
        initialize_random_snakes(p, a, game, rng_state);
    } else {
        for (int s = 0; s < p.num_snakes; ++s) {
            int body_len = init.snake_body_lengths[s];
            if (body_len > p.max_body_len) {
                body_len = p.max_body_len;
            }
            int si = snake_index(p, game, s);
            a.body_size[si] = body_len;
            for (int i = 0; i < body_len; ++i) {
                int src = s * p.max_init_body_length * 2 + i * 2;
                int x = init.snake_bodies[src];
                int y = init.snake_bodies[src + 1];
                int bi = body_index(p, game, s, i);
                a.body_x[bi] = x;
                a.body_y[bi] = y;
                if (in_bounds(p, x, y)) {
                    a.board[board_index(p, game, s, cell_index(p, x, y))] = 1;
                }
            }
        }
    }

    a.food_count[game] = 0;
    if (p.num_init_food == -2) {
        return;
    }
    if (p.num_init_food == -1) {
        initialize_random_food(p, a, game, p.init_turns_played, rng_state);
        return;
    }
    int food_count = p.num_init_food;
    if (food_count > p.max_food) {
        food_count = p.max_food;
    }
    for (int i = 0; i < food_count; ++i) {
        int src = i * 2;
        int fidx = food_index(p, game, i);
        a.food_x[fidx] = init.food_spawns[src];
        a.food_y[fidx] = init.food_spawns[src + 1];
        a.food_turn[fidx] = init.food_spawn_turn_values != nullptr
            ? init.food_spawn_turn_values[i]
            : p.init_turns_played;
    }
    a.food_count[game] = food_count;
}

__device__ int legal_action_mask(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int snake_id
) {
    int si = snake_index(p, game, snake_id);
    if (!a.alive[si] || a.body_size[si] <= 0) {
        return 0;
    }
    int head_idx = body_index(p, game, snake_id, 0);
    int head_x = a.body_x[head_idx];
    int head_y = a.body_y[head_idx];
    int mask = 0;

    for (int move = 0; move < 4; ++move) {
        int nx;
        int ny;
        new_position(p, head_x, head_y, move, &nx, &ny);
        if (!in_bounds(p, nx, ny)) {
            continue;
        }
        int new_cell = cell_index(p, nx, ny);
        bool collision = false;
        for (int other = 0; other < p.num_snakes; ++other) {
            int oi = snake_index(p, game, other);
            if (!a.alive[oi]) {
                continue;
            }
            if (a.board[board_index(p, game, other, new_cell)]) {
                int other_body_size = a.body_size[oi];
                int tail_idx = body_index(p, game, other, other_body_size - 1);
                bool is_tail = a.body_x[tail_idx] == nx && a.body_y[tail_idx] == ny;
                if (!is_tail || a.length[oi] > other_body_size) {
                    collision = true;
                    break;
                }
            }
        }
        if (collision) {
            continue;
        }

        bool is_food = false;
        int food_count = a.food_count[game];
        for (int i = 0; i < food_count; ++i) {
            int fidx = food_index(p, game, i);
            if (a.food_x[fidx] == nx && a.food_y[fidx] == ny) {
                is_food = true;
                break;
            }
        }

        int cur_health = a.health[si];
        if (!is_food && cur_health == 1) {
            continue;
        }
        if (!is_food &&
            a.hazards[game_cell_index(p, game, new_cell)] &&
            cur_health <= p.hazard_damage + 1) {
            continue;
        }
        mask |= (1 << move);
    }
    return mask;
}

__device__ int popcount4(int mask) {
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        count += (mask >> i) & 1;
    }
    return count;
}

__device__ int choose_action_from_mask(unsigned long long* rng_state, int mask) {
    int count = popcount4(mask);
    if (count == 0) {
        return 0;
    }
    int chosen = random_mod(rng_state, count);
    for (int move = 0; move < 4; ++move) {
        if ((mask >> move) & 1) {
            if (chosen == 0) {
                return move;
            }
            chosen--;
        }
    }
    return 0;
}

__device__ void compact_food_after_eating(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    unsigned char* scratch_game
) {
    int old_count = a.food_count[game];
    int write = 0;
    for (int read = 0; read < old_count; ++read) {
        if (scratch_game[read]) {
            continue;
        }
        if (write != read) {
            int src = food_index(p, game, read);
            int dst = food_index(p, game, write);
            a.food_x[dst] = a.food_x[src];
            a.food_y[dst] = a.food_y[src];
            a.food_turn[dst] = a.food_turn[src];
        }
        write++;
    }
    a.food_count[game] = write;
}

__device__ void place_food_randomly(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    unsigned long long* rng_state,
    unsigned char* scratch_game
) {
    int num_food = a.food_count[game];
    int n_to_place = 0;
    if (num_food < p.min_food) {
        n_to_place = p.min_food - num_food;
    }
    if (random_mod(rng_state, 100) < p.food_spawn_chance) {
        n_to_place++;
    }
    if (n_to_place <= 0) {
        return;
    }

    for (int cell = 0; cell < p.cells; ++cell) {
        scratch_game[cell] = 1;
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        if (!a.alive[si]) {
            continue;
        }
        int body_size = a.body_size[si];
        for (int i = 0; i < body_size; ++i) {
            int bi = body_index(p, game, s, i);
            int x = a.body_x[bi];
            int y = a.body_y[bi];
            if (in_bounds(p, x, y)) {
                scratch_game[cell_index(p, x, y)] = 0;
            }
        }
        if (a.length[si] > body_size && body_size > 0) {
            int tail_idx = body_index(p, game, s, body_size - 1);
            int tx = a.body_x[tail_idx];
            int ty = a.body_y[tail_idx];
            if (in_bounds(p, tx, ty)) {
                scratch_game[cell_index(p, tx, ty)] = 0;
            }
        }
        int head_idx = body_index(p, game, s, 0);
        int hx = a.body_x[head_idx];
        int hy = a.body_y[head_idx];
        for (int move = 0; move < 4; ++move) {
            int nx;
            int ny;
            new_position(p, hx, hy, move, &nx, &ny);
            if (in_bounds(p, nx, ny)) {
                scratch_game[cell_index(p, nx, ny)] = 0;
            }
        }
    }

    int existing_food = a.food_count[game];
    for (int i = 0; i < existing_food; ++i) {
        int fidx = food_index(p, game, i);
        int fx = a.food_x[fidx];
        int fy = a.food_y[fidx];
        if (in_bounds(p, fx, fy)) {
            scratch_game[cell_index(p, fx, fy)] = 0;
        }
    }

    int num_spawns = 0;
    for (int cell = 0; cell < p.cells; ++cell) {
        num_spawns += scratch_game[cell] ? 1 : 0;
    }
    if (num_spawns < n_to_place) {
        n_to_place = num_spawns;
    }
    if (num_spawns == 0) {
        return;
    }

    for (int i = 0; i < n_to_place && a.food_count[game] < p.max_food; ++i) {
        int rng = random_mod(rng_state, num_spawns);
        int counter = 0;
        for (int cell = 0; cell < p.cells; ++cell) {
            if (!scratch_game[cell]) {
                continue;
            }
            if (counter == rng) {
                int food_count = a.food_count[game];
                int fidx = food_index(p, game, food_count);
                a.food_x[fidx] = cell % p.w;
                a.food_y[fidx] = cell / p.w;
                a.food_turn[fidx] = turn + 1;
                a.food_count[game] = food_count + 1;
                scratch_game[cell] = 0;
                num_spawns--;
                break;
            }
            counter++;
        }
        if (num_spawns == 0) {
            break;
        }
    }
}

__device__ void maybe_update_hazards(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    unsigned long long* rng_state
) {
    if (!p.royale) {
        return;
    }
    if (turn < p.shrink_n_turns) {
        return;
    }
    if (p.shrink_n_turns <= 0 || turn % p.shrink_n_turns != 0) {
        return;
    }

    int min_x = p.w;
    int min_y = p.h;
    int max_x = -1;
    int max_y = -1;
    for (int y = 0; y < p.h; ++y) {
        for (int x = 0; x < p.w; ++x) {
            int cell = cell_index(p, x, y);
            if (!a.hazards[game_cell_index(p, game, cell)]) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    if (max_x < 0 || max_y < 0) {
        return;
    }

    int rng = random_mod(rng_state, 4);
    if (rng == 0) {
        for (int y = 0; y < p.h; ++y) {
            a.hazards[game_cell_index(p, game, cell_index(p, min_x, y))] = 1;
        }
    } else if (rng == 1) {
        for (int y = 0; y < p.h; ++y) {
            a.hazards[game_cell_index(p, game, cell_index(p, max_x, y))] = 1;
        }
    } else if (rng == 2) {
        for (int x = 0; x < p.w; ++x) {
            a.hazards[game_cell_index(p, game, cell_index(p, x, min_y))] = 1;
        }
    } else {
        for (int x = 0; x < p.w; ++x) {
            a.hazards[game_cell_index(p, game, cell_index(p, x, max_y))] = 1;
        }
    }
}

__device__ int step_game(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    const int* actions,
    unsigned long long* rng_state
) {
    unsigned char* scratch_game = a.scratch + game * p.cells;
    int food_count = a.food_count[game];
    for (int i = 0; i < food_count; ++i) {
        scratch_game[i] = 0;
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        if (!a.alive[si] || a.body_size[si] <= 0) {
            continue;
        }
        int old_head = body_index(p, game, s, 0);
        int nx;
        int ny;
        new_position(p, a.body_x[old_head], a.body_y[old_head], actions[s], &nx, &ny);

        int body_size = a.body_size[si];
        int write_limit = body_size;
        if (write_limit >= p.max_body_len) {
            write_limit = p.max_body_len - 1;
        }
        for (int i = write_limit; i > 0; --i) {
            int dst = body_index(p, game, s, i);
            int src = body_index(p, game, s, i - 1);
            a.body_x[dst] = a.body_x[src];
            a.body_y[dst] = a.body_y[src];
        }
        int new_head = body_index(p, game, s, 0);
        a.body_x[new_head] = nx;
        a.body_y[new_head] = ny;
        if (body_size < p.max_body_len) {
            body_size++;
        }
        a.body_size[si] = body_size;

        if (a.length[si] < body_size) {
            int tail = body_index(p, game, s, body_size - 1);
            int tx = a.body_x[tail];
            int ty = a.body_y[tail];
            if (in_bounds(p, tx, ty)) {
                a.board[board_index(p, game, s, cell_index(p, tx, ty))] = 0;
            }
            a.body_size[si] = body_size - 1;
            body_size--;
        }

        int health = a.health[si] - 1;
        if (p.royale && in_bounds(p, nx, ny) &&
            a.hazards[game_cell_index(p, game, cell_index(p, nx, ny))]) {
            health -= p.hazard_damage;
        }
        if (health < 0) {
            health = 0;
        }
        a.health[si] = health;

        for (int f = 0; f < food_count; ++f) {
            int fidx = food_index(p, game, f);
            if (a.food_x[fidx] == nx && a.food_y[fidx] == ny) {
                a.health[si] = a.max_health[si];
                a.length[si] += 1;
                scratch_game[f] = 1;
            }
        }
    }

    compact_food_after_eating(p, a, game, scratch_game);
    place_food_randomly(p, a, game, turn, rng_state, scratch_game);

    int death_cause[HISSS_CUDA_MAX_SNAKES];
    int death_killer[HISSS_CUDA_MAX_SNAKES];
    for (int s = 0; s < HISSS_CUDA_MAX_SNAKES; ++s) {
        death_cause[s] = 0;
        death_killer[s] = -1;
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        if (!a.alive[si] || a.body_size[si] <= 0) {
            continue;
        }
        int head = body_index(p, game, s, 0);
        int hx = a.body_x[head];
        int hy = a.body_y[head];
        if (!in_bounds(p, hx, hy)) {
            death_cause[s] = 1;
            continue;
        }
        if (a.health[si] <= 0) {
            death_cause[s] = 2;
            continue;
        }
        int head_cell = cell_index(p, hx, hy);
        if (a.board[board_index(p, game, s, head_cell)]) {
            death_cause[s] = 3;
            continue;
        }
        for (int other = 0; other < p.num_snakes; ++other) {
            if (other == s) {
                continue;
            }
            int oi = snake_index(p, game, other);
            if (!a.alive[oi]) {
                continue;
            }
            if (a.board[board_index(p, game, other, head_cell)]) {
                death_cause[s] = 4;
                death_killer[s] = other;
                break;
            }
            int other_head = body_index(p, game, other, 0);
            if (hx == a.body_x[other_head] && hy == a.body_y[other_head] &&
                a.length[si] <= a.length[oi]) {
                death_cause[s] = 5;
                death_killer[s] = other;
                break;
            }
        }
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        if (death_cause[s] == 0) {
            continue;
        }
        int si = snake_index(p, game, s);
        a.alive[si] = 0;
        a.death_cause[si] = death_cause[s];
        a.death_turn[si] = turn;
        a.killer_id[si] = death_killer[s];
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        if (!a.alive[si] || a.body_size[si] <= 0) {
            continue;
        }
        int head = body_index(p, game, s, 0);
        int hx = a.body_x[head];
        int hy = a.body_y[head];
        if (in_bounds(p, hx, hy)) {
            a.board[board_index(p, game, s, cell_index(p, hx, hy))] = 1;
        }
    }

    turn += 1;
    maybe_update_hazards(p, a, game, turn, rng_state);
    return turn;
}

__device__ int compute_at_turn(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int* action_masks,
    int* sole_player
) {
    int at_turn = 0;
    *sole_player = -1;
    for (int s = 0; s < p.num_snakes; ++s) {
        int mask = legal_action_mask(p, a, game, s);
        action_masks[s] = mask;
        if (mask != 0) {
            at_turn++;
            *sole_player = s;
        }
    }
    return at_turn;
}

__device__ int winner_from_state(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int at_turn,
    int sole_at_turn
) {
    if (at_turn == 1) {
        return sole_at_turn;
    }
    int alive_count = 0;
    int winner = -1;
    for (int s = 0; s < p.num_snakes; ++s) {
        int si = snake_index(p, game, s);
        if (a.alive[si]) {
            alive_count++;
            winner = s;
        }
    }
    return alive_count == 1 ? winner : -1;
}

__global__ void simulate_random_rollouts_kernel(
    SimParams p,
    InitArrays init,
    DeviceArrays a,
    unsigned long long seed
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= p.num_games) {
        return;
    }

    unsigned long long rng_state =
        seed ^ (0xD1B54A32D192ED03ull * static_cast<unsigned long long>(game + 1));
    rng_state = splitmix64_next(&rng_state);

    initialize_game(p, init, a, game, &rng_state);

    int turn = p.init_turns_played;
    bool terminal = false;
    int winner = -1;
    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int actions[HISSS_CUDA_MAX_SNAKES];

    for (int step = 0; step < p.max_turns; ++step) {
        int sole_at_turn = -1;
        int at_turn = compute_at_turn(p, a, game, action_masks, &sole_at_turn);
        if ((p.num_snakes == 1 && at_turn == 0) ||
            (p.num_snakes > 1 && at_turn <= 1)) {
            terminal = true;
            winner = winner_from_state(p, a, game, at_turn, sole_at_turn);
            break;
        }

        for (int s = 0; s < p.num_snakes; ++s) {
            actions[s] = choose_action_from_mask(&rng_state, action_masks[s]);
        }
        turn = step_game(p, a, game, turn, actions, &rng_state);
    }

    if (!terminal) {
        int sole_at_turn = -1;
        int at_turn = compute_at_turn(p, a, game, action_masks, &sole_at_turn);
        if ((p.num_snakes == 1 && at_turn == 0) ||
            (p.num_snakes > 1 && at_turn <= 1)) {
            terminal = true;
            winner = winner_from_state(p, a, game, at_turn, sole_at_turn);
        }
    }

    a.out_turns_played[game] = turn;
    a.out_terminal[game] = terminal ? 1 : 0;
    a.out_winner[game] = winner;
}

}  // namespace

bool hisss_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        set_cuda_error("cudaGetDeviceCount", err);
        return false;
    }
    if (device_count <= 0) {
        set_last_error("No CUDA devices available");
        return false;
    }
    return true;
}

const char* hisss_cuda_last_error() {
    return g_hisss_cuda_error.c_str();
}

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
) {
    g_hisss_cuda_error.clear();
    if (w <= 0 || h <= 0 || num_snakes <= 0 || num_games <= 0 || max_turns < 0) {
        set_last_error("invalid CUDA rollout dimensions");
        return 1;
    }
    if (num_snakes > HISSS_CUDA_MAX_SNAKES) {
        std::ostringstream oss;
        oss << "CUDA rollout supports at most " << HISSS_CUDA_MAX_SNAKES
            << " snakes per game";
        set_last_error(oss.str());
        return 1;
    }
    if (spawn_snakes_randomly && num_snakes > 8) {
        set_last_error("random CUDA spawning supports at most 8 snakes");
        return 1;
    }
    if (!snake_alive || !snake_health || !snake_len || !max_health || !init_hazards) {
        set_last_error("missing required CUDA rollout input arrays");
        return 1;
    }
    if (!out_turns_played || !out_terminal || !out_winner || !out_alive ||
        !out_lengths || !out_health || !out_death_cause || !out_death_turn ||
        !out_killer_id) {
        set_last_error("missing required CUDA rollout output arrays");
        return 1;
    }
    if (!spawn_snakes_randomly && (!snake_body_lengths || !snake_bodies ||
                                  max_init_body_length <= 0)) {
        set_last_error("fixed CUDA spawning requires snake body arrays");
        return 1;
    }
    if (num_init_food > 0 && !food_spawns) {
        set_last_error("fixed CUDA food spawning requires food_spawns");
        return 1;
    }

    int cells = w * h;
    int max_init_len = 0;
    for (int s = 0; s < num_snakes; ++s) {
        max_init_len = std::max(max_init_len, snake_len[s]);
    }
    int max_body_len = std::max({cells + 2, max_init_len + 1, max_init_body_length + 1});
    int max_food = cells;

    SimParams p;
    p.w = w;
    p.h = h;
    p.cells = cells;
    p.num_snakes = num_snakes;
    p.min_food = min_food;
    p.food_spawn_chance = food_spawn_chance;
    p.init_turns_played = init_turns_played;
    p.spawn_snakes_randomly = spawn_snakes_randomly;
    p.max_init_body_length = max_init_body_length;
    p.num_init_food = num_init_food;
    p.wrapped = wrapped;
    p.royale = royale;
    p.shrink_n_turns = shrink_n_turns;
    p.hazard_damage = hazard_damage;
    p.num_games = num_games;
    p.max_turns = max_turns;
    p.max_body_len = max_body_len;
    p.max_food = max_food;
    p.duel_probability_ppm = 0;

    InitArrays init{};
    DeviceArrays a{};

    bool ok = true;
    ok = ok && device_alloc_copy(&init.snake_alive, snake_alive, num_snakes, "snake_alive");
    ok = ok && device_alloc_copy(&init.snake_health, snake_health, num_snakes, "snake_health");
    ok = ok && device_alloc_copy(&init.snake_len, snake_len, num_snakes, "snake_len");
    ok = ok && device_alloc_copy(&init.max_health, max_health, num_snakes, "max_health");
    ok = ok && device_alloc_copy(&init.init_hazards, init_hazards, cells, "init_hazards");
    if (!spawn_snakes_randomly) {
        size_t body_count = static_cast<size_t>(num_snakes) *
            static_cast<size_t>(max_init_body_length) * 2u;
        ok = ok && device_alloc_copy(
            &init.snake_body_lengths,
            snake_body_lengths,
            num_snakes,
            "snake_body_lengths"
        );
        ok = ok && device_alloc_copy(&init.snake_bodies, snake_bodies, body_count, "snake_bodies");
    }
    if (num_init_food > 0) {
        ok = ok && device_alloc_copy(
            &init.food_spawns,
            food_spawns,
            static_cast<size_t>(num_init_food) * 2u,
            "food_spawns"
        );
        if (food_spawn_turn_values != nullptr) {
            ok = ok && device_alloc_copy(
                &init.food_spawn_turn_values,
                food_spawn_turn_values,
                num_init_food,
                "food_spawn_turn_values"
            );
        }
    }
    if (!ok) {
        free_init_arrays(init);
        return 1;
    }

    size_t games_snakes = static_cast<size_t>(num_games) * static_cast<size_t>(num_snakes);
    size_t body_entries = games_snakes * static_cast<size_t>(max_body_len);
    size_t board_entries = games_snakes * static_cast<size_t>(cells);
    size_t food_entries = static_cast<size_t>(num_games) * static_cast<size_t>(max_food);
    size_t game_cells = static_cast<size_t>(num_games) * static_cast<size_t>(cells);

    ok = ok && device_alloc(&a.body_x, body_entries, "body_x");
    ok = ok && device_alloc(&a.body_y, body_entries, "body_y");
    ok = ok && device_alloc(&a.body_size, games_snakes, "body_size");
    ok = ok && device_alloc(&a.board, board_entries, "board");
    ok = ok && device_alloc(&a.food_x, food_entries, "food_x");
    ok = ok && device_alloc(&a.food_y, food_entries, "food_y");
    ok = ok && device_alloc(&a.food_turn, food_entries, "food_turn");
    ok = ok && device_alloc(&a.food_count, num_games, "food_count");
    ok = ok && device_alloc(&a.scratch, game_cells, "scratch");
    ok = ok && device_alloc(&a.hazards, game_cells, "hazards");
    ok = ok && device_alloc(&a.alive, games_snakes, "alive");
    ok = ok && device_alloc(&a.health, games_snakes, "health");
    ok = ok && device_alloc(&a.length, games_snakes, "length");
    ok = ok && device_alloc(&a.max_health, games_snakes, "max_health");
    ok = ok && device_alloc(&a.death_cause, games_snakes, "death_cause");
    ok = ok && device_alloc(&a.death_turn, games_snakes, "death_turn");
    ok = ok && device_alloc(&a.killer_id, games_snakes, "killer_id");
    ok = ok && device_alloc(&a.out_turns_played, num_games, "out_turns_played");
    ok = ok && device_alloc(&a.out_terminal, num_games, "out_terminal");
    ok = ok && device_alloc(&a.out_winner, num_games, "out_winner");

    if (!ok) {
        free_init_arrays(init);
        free_device_arrays(a);
        return 1;
    }

    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (num_games + block_size - 1) / block_size;
    simulate_random_rollouts_kernel<<<grid_size, block_size>>>(p, init, a, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("simulate_random_rollouts_kernel launch", err);
        free_init_arrays(init);
        free_device_arrays(a);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("simulate_random_rollouts_kernel execution", err);
        free_init_arrays(init);
        free_device_arrays(a);
        return 1;
    }

    ok = true;
    err = cudaMemcpy(out_turns_played, a.out_turns_played, num_games * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy out_turns_played", err); ok = false; }
    err = cudaMemcpy(out_terminal, a.out_terminal, num_games * sizeof(bool), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy out_terminal", err); ok = false; }
    err = cudaMemcpy(out_winner, a.out_winner, num_games * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy out_winner", err); ok = false; }
    err = cudaMemcpy(out_alive, a.alive, games_snakes * sizeof(bool), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy alive", err); ok = false; }
    err = cudaMemcpy(out_lengths, a.length, games_snakes * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy lengths", err); ok = false; }
    err = cudaMemcpy(out_health, a.health, games_snakes * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy health", err); ok = false; }
    err = cudaMemcpy(out_death_cause, a.death_cause, games_snakes * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy death_cause", err); ok = false; }
    err = cudaMemcpy(out_death_turn, a.death_turn, games_snakes * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy death_turn", err); ok = false; }
    err = cudaMemcpy(out_killer_id, a.killer_id, games_snakes * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_error("copy killer_id", err); ok = false; }

    free_init_arrays(init);
    free_device_arrays(a);
    return ok ? 0 : 1;
}

namespace {

constexpr int BLACKOUT_W = 15;
constexpr int BLACKOUT_H = 15;
constexpr int BLACKOUT_SNAKES = 4;
constexpr int BLACKOUT_VIEW_RADIUS = 5;
// Channels 0..8 are the fogged policy observation. Channels 9..13 are
// privileged (unfogged) planes for an asymmetric critic: they exist only at
// training time and the actor/inference path slices them away, so fog
// semantics of the deployed policy are unchanged.
constexpr int BLACKOUT_POLICY_CHANNELS = 9;
constexpr int BLACKOUT_PRIV_FOOD = 9;        // all food, no visibility gate
constexpr int BLACKOUT_PRIV_ENEMY_BODY = 10; // per-enemy decay (length - i) / 10
constexpr int BLACKOUT_PRIV_ENEMY_HEAD = 11; // 1.0 at each enemy head
constexpr int BLACKOUT_PRIV_ENEMY_HEALTH = 12; // health / 100 on body cells
constexpr int BLACKOUT_PRIV_ENEMY_TAIL = 13; // 1.0 at each enemy tail
constexpr int BLACKOUT_OBS_CHANNELS = 14;
constexpr int BLACKOUT_OBS_XY = 29;
constexpr int BLACKOUT_OBS_PLANE = BLACKOUT_OBS_XY * BLACKOUT_OBS_XY;
constexpr int BLACKOUT_OBS_SIZE = BLACKOUT_OBS_CHANNELS * BLACKOUT_OBS_XY * BLACKOUT_OBS_XY;
constexpr int BLACKOUT_HEURISTIC_FORAGER = 0;
constexpr int BLACKOUT_HEURISTIC_HUNTER = 1;
constexpr int BLACKOUT_HEURISTIC_TERRITORIAL = 2;
constexpr int BLACKOUT_HEURISTIC_EDGE_TRAPPER = 3;
constexpr int BLACKOUT_HEURISTIC_SURVIVOR = 4;
constexpr int BLACKOUT_HEURISTIC_SNAKE25 = 5;
constexpr int BLACKOUT_HEURISTIC_SNAKE25_INTERCEPTOR = 6;
constexpr int BLACKOUT_HEURISTIC_SNAKE25_DENIER = 7;
constexpr int BLACKOUT_HEURISTIC_SNAKE25_DUELIST = 8;
constexpr int BLACKOUT_HEURISTIC_COUNT = 9;

struct BlackoutVecEnv {
    SimParams p;
    InitArrays init;
    DeviceArrays a;
    int num_envs;
    int* d_turns;
    unsigned long long* d_rng_states;
    int* d_actions;
    float* d_obs;
    float* d_rewards;
    unsigned char* d_done;
    unsigned char* d_legal_mask;
    int* d_best_action_masks;
    unsigned char* d_best_scratch;
};

__device__ bool blackout_cell_blocked(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int x,
    int y
) {
    if (!in_bounds(p, x, y)) {
        return true;
    }
    int cell = cell_index(p, x, y);
    for (int snake = 0; snake < p.num_snakes; ++snake) {
        int si = snake_index(p, game, snake);
        if (!a.alive[si] || !a.board[board_index(p, game, snake, cell)]) {
            continue;
        }

        int body_size = a.body_size[si];
        bool vacating_tail = false;
        if (body_size > 0 && a.length[si] <= body_size) {
            int tail = body_index(p, game, snake, body_size - 1);
            vacating_tail = a.body_x[tail] == x && a.body_y[tail] == y;
        }
        if (!vacating_tail) {
            return true;
        }
    }
    return false;
}

__device__ int blackout_reachable_space(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int start_x,
    int start_y,
    unsigned char* queue
) {
    if (!in_bounds(p, start_x, start_y)) {
        return 0;
    }

    // 15x15 cells fit in eight 32-bit words. The caller supplies a compact
    // per-work-item byte queue (cell ids are in [0, 224]).
    unsigned int visited[8];
    for (int i = 0; i < 8; ++i) {
        visited[i] = 0u;
    }
    int start_cell = cell_index(p, start_x, start_y);
    visited[start_cell >> 5] |= 1u << (start_cell & 31);
    queue[0] = static_cast<unsigned char>(start_cell);
    int read = 0;
    int write = 1;

    while (read < write) {
        int cell = static_cast<int>(queue[read++]);
        int x = cell % p.w;
        int y = cell / p.w;
        for (int move = 0; move < 4; ++move) {
            int nx;
            int ny;
            new_position(p, x, y, move, &nx, &ny);
            if (blackout_cell_blocked(p, a, game, nx, ny)) {
                continue;
            }
            int next_cell = cell_index(p, nx, ny);
            unsigned int bit = 1u << (next_cell & 31);
            if (visited[next_cell >> 5] & bit) {
                continue;
            }
            visited[next_cell >> 5] |= bit;
            queue[write++] = static_cast<unsigned char>(next_cell);
        }
    }
    return write;
}

__device__ bool blackout_observed_cell_blocked(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int observer,
    int observer_x,
    int observer_y,
    int x,
    int y
) {
    if (!in_bounds(p, x, y)) {
        return true;
    }
    int cell = cell_index(p, x, y);
    for (int snake = 0; snake < p.num_snakes; ++snake) {
        int si = snake_index(p, game, snake);
        if (!a.alive[si] || !a.board[board_index(p, game, snake, cell)]) {
            continue;
        }
        if (snake != observer &&
            abs(x - observer_x) + abs(y - observer_y) > BLACKOUT_VIEW_RADIUS) {
            // Unknown Blackout cells are open from the policy's perspective.
            continue;
        }
        int body_size = a.body_size[si];
        bool vacating_tail = false;
        if (body_size > 0 && a.length[si] <= body_size) {
            int tail = body_index(p, game, snake, body_size - 1);
            vacating_tail = a.body_x[tail] == x && a.body_y[tail] == y;
        }
        if (!vacating_tail) {
            return true;
        }
    }
    return false;
}

__device__ int blackout_observed_reachable_space(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int observer,
    int observer_x,
    int observer_y,
    int start_x,
    int start_y,
    unsigned char* queue
) {
    if (!in_bounds(p, start_x, start_y)) {
        return 0;
    }
    unsigned int visited[8];
    for (int i = 0; i < 8; ++i) {
        visited[i] = 0u;
    }
    int start_cell = cell_index(p, start_x, start_y);
    visited[start_cell >> 5] |= 1u << (start_cell & 31);
    queue[0] = static_cast<unsigned char>(start_cell);
    int read = 0;
    int write = 1;
    while (read < write) {
        int cell = static_cast<int>(queue[read++]);
        int x = cell % p.w;
        int y = cell / p.w;
        for (int move = 0; move < 4; ++move) {
            int nx;
            int ny;
            new_position(p, x, y, move, &nx, &ny);
            if (blackout_observed_cell_blocked(
                    p,
                    a,
                    game,
                    observer,
                    observer_x,
                    observer_y,
                    nx,
                    ny
                )) {
                continue;
            }
            int next_cell = cell_index(p, nx, ny);
            unsigned int bit = 1u << (next_cell & 31);
            if (visited[next_cell >> 5] & bit) {
                continue;
            }
            visited[next_cell >> 5] |= bit;
            queue[write++] = static_cast<unsigned char>(next_cell);
        }
    }
    return write;
}

__device__ int blackout_open_neighbors(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int x,
    int y
) {
    int count = 0;
    for (int move = 0; move < 4; ++move) {
        int nx;
        int ny;
        new_position(p, x, y, move, &nx, &ny);
        if (!blackout_cell_blocked(p, a, game, nx, ny)) {
            count++;
        }
    }
    return count;
}

__device__ int blackout_wall_distance(const SimParams& p, int x, int y) {
    int result = x;
    if (y < result) {
        result = y;
    }
    int right = p.w - 1 - x;
    if (right < result) {
        result = right;
    }
    int top = p.h - 1 - y;
    if (top < result) {
        result = top;
    }
    return result;
}

__device__ int blackout_visible_food_distance(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    int head_x,
    int head_y,
    int x,
    int y
) {
    int nearest = p.w + p.h + 1;
    int count = a.food_count[game];
    for (int food = 0; food < count; ++food) {
        int fi = food_index(p, game, food);
        int fx = a.food_x[fi];
        int fy = a.food_y[fi];
        bool visible = abs(fx - head_x) + abs(fy - head_y) <= BLACKOUT_VIEW_RADIUS ||
                       a.food_turn[fi] == turn;
        if (!visible) {
            continue;
        }
        int distance = abs(fx - x) + abs(fy - y);
        if (distance < nearest) {
            nearest = distance;
        }
    }
    return nearest;
}

__device__ int blackout_action_for_delta(int dx, int dy) {
    if (dx == 1 && dy == 0) {
        return 1;
    }
    if (dx == 0 && dy == -1) {
        return 2;
    }
    if (dx == -1 && dy == 0) {
        return 3;
    }
    return 0;
}

__device__ int blackout_best_action(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int snake,
    int turn,
    const int* action_masks,
    unsigned char* scratch
) {
    int mask = action_masks[snake];
    if (mask == 0) {
        return 0;
    }

    int si = snake_index(p, game, snake);
    int head = body_index(p, game, snake, 0);
    int head_x = a.body_x[head];
    int head_y = a.body_y[head];
    int snake_length = a.length[si];
    int health = a.health[si];
    int tie_start = (game + snake + turn) & 3;
    int best_move = 0;
    float best_score = -1.0e30f;
    int best_tie_rank = 5;

    for (int move = 0; move < 4; ++move) {
        if (((mask >> move) & 1) == 0) {
            continue;
        }
        int nx;
        int ny;
        new_position(p, head_x, head_y, move, &nx, &ny);

        int reachable = blackout_reachable_space(p, a, game, nx, ny, scratch);
        int exits = blackout_open_neighbors(p, a, game, nx, ny);
        int wall = blackout_wall_distance(p, nx, ny);
        float score = static_cast<float>(reachable) +
                      4.0f * static_cast<float>(exits) +
                      0.5f * static_cast<float>(wall);

        int minimum_space = snake_length + 2;
        if (minimum_space < 8) {
            minimum_space = 8;
        }
        if (reachable < minimum_space) {
            score -= 24.0f * static_cast<float>(minimum_space - reachable);
        }
        if (exits <= 1) {
            score -= 18.0f;
        }

        int food_distance = blackout_visible_food_distance(
            p,
            a,
            game,
            turn,
            head_x,
            head_y,
            nx,
            ny
        );
        if (food_distance <= p.w + p.h) {
            float food_weight = 0.75f;
            if (health <= 25) {
                food_weight = 9.0f;
            } else if (health <= 50) {
                food_weight = 5.0f;
            } else if (health <= 75) {
                food_weight = 2.0f;
            }
            score += food_weight * static_cast<float>(p.w + p.h - food_distance);
            if (food_distance == 0) {
                score += 8.0f * food_weight;
            }
        } else if (a.body_size[si] > 1) {
            int tail = body_index(p, game, snake, a.body_size[si] - 1);
            int tail_distance = abs(a.body_x[tail] - nx) + abs(a.body_y[tail] - ny);
            score += 0.25f * static_cast<float>(p.w + p.h - tail_distance);
        }

        for (int other = 0; other < p.num_snakes; ++other) {
            if (other == snake || action_masks[other] == 0) {
                continue;
            }
            int oi = snake_index(p, game, other);
            if (!a.alive[oi] || a.body_size[oi] <= 0) {
                continue;
            }
            int other_head = body_index(p, game, other, 0);
            int ox = a.body_x[other_head];
            int oy = a.body_y[other_head];
            int dx = nx - ox;
            int dy = ny - oy;
            if (abs(dx) + abs(dy) != 1) {
                continue;
            }
            int enemy_move = blackout_action_for_delta(dx, dy);
            if (((action_masks[other] >> enemy_move) & 1) == 0) {
                continue;
            }
            if (a.length[oi] >= snake_length) {
                score -= 1000.0f;
            } else {
                score += 18.0f;
            }
        }

        int tie_rank = (move - tie_start + 4) & 3;
        if (score > best_score || (score == best_score && tie_rank < best_tie_rank)) {
            best_score = score;
            best_move = move;
            best_tie_rank = tie_rank;
        }
    }
    return best_move;
}

__device__ bool blackout_head_visible(
    int self_x,
    int self_y,
    int other_x,
    int other_y
) {
    return abs(other_x - self_x) + abs(other_y - self_y) <=
           BLACKOUT_VIEW_RADIUS;
}

__device__ int blackout_previous_action(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int snake
) {
    int si = snake_index(p, game, snake);
    int body_size = a.body_size[si];
    if (body_size <= 1) {
        return -1;
    }
    int head = body_index(p, game, snake, 0);
    int head_x = a.body_x[head];
    int head_y = a.body_y[head];
    for (int body_pos = 1; body_pos < body_size; ++body_pos) {
        int neck = body_index(p, game, snake, body_pos);
        int dx = head_x - a.body_x[neck];
        int dy = head_y - a.body_y[neck];
        if (abs(dx) + abs(dy) == 1) {
            return blackout_action_for_delta(dx, dy);
        }
    }
    return -1;
}

__device__ float blackout_profile_score(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int snake,
    int turn,
    int profile,
    int move,
    const int* action_masks,
    unsigned char* scratch
) {
    int si = snake_index(p, game, snake);
    int head = body_index(p, game, snake, 0);
    int head_x = a.body_x[head];
    int head_y = a.body_y[head];
    int nx;
    int ny;
    new_position(p, head_x, head_y, move, &nx, &ny);

    int snake_length = a.length[si];
    int health = a.health[si];
    int exits = blackout_open_neighbors(p, a, game, nx, ny);
    int wall = blackout_wall_distance(p, nx, ny);
    int food_distance = blackout_visible_food_distance(
        p, a, game, turn, head_x, head_y, nx, ny
    );
    bool food_visible = food_distance <= p.w + p.h;
    bool eats = food_distance == 0;
    int previous_action = blackout_previous_action(p, a, game, snake);

    int nearest_enemy = p.w + p.h + 1;
    int nearest_prey = p.w + p.h + 1;
    int nearest_danger = p.w + p.h + 1;
    int nearest_edge_enemy = p.w + p.h + 1;
    bool losing_contest = false;
    bool winning_contest = false;
    for (int other = 0; other < p.num_snakes; ++other) {
        if (other == snake || action_masks[other] == 0) {
            continue;
        }
        int oi = snake_index(p, game, other);
        if (!a.alive[oi] || a.body_size[oi] <= 0) {
            continue;
        }
        int other_head = body_index(p, game, other, 0);
        int ox = a.body_x[other_head];
        int oy = a.body_y[other_head];
        if (!blackout_head_visible(head_x, head_y, ox, oy)) {
            continue;
        }
        int distance = abs(nx - ox) + abs(ny - oy);
        if (distance < nearest_enemy) {
            nearest_enemy = distance;
        }
        if (a.length[oi] < snake_length) {
            if (distance < nearest_prey) {
                nearest_prey = distance;
            }
        } else if (distance < nearest_danger) {
            nearest_danger = distance;
        }
        if (blackout_wall_distance(p, ox, oy) <= 1 &&
            distance < nearest_edge_enemy) {
            nearest_edge_enemy = distance;
        }
        if (distance == 1) {
            int enemy_move = blackout_action_for_delta(nx - ox, ny - oy);
            if ((action_masks[other] >> enemy_move) & 1) {
                if (a.length[oi] >= snake_length) {
                    losing_contest = true;
                } else {
                    winning_contest = true;
                }
            }
        }
    }

    int own_tail_distance = p.w + p.h;
    if (a.body_size[si] > 1) {
        int tail = body_index(p, game, snake, a.body_size[si] - 1);
        own_tail_distance =
            abs(a.body_x[tail] - nx) + abs(a.body_y[tail] - ny);
    }
    bool hazard = a.hazards[
        game_cell_index(p, game, cell_index(p, nx, ny))
    ] != 0;
    float score = 0.0f;

    if (profile == BLACKOUT_HEURISTIC_FORAGER) {
        // Mirrors the 34/35/40 cluster: 84-89% food-progress and unusually
        // persistent lines, accepting more edge and corridor exposure.
        float hunger = health <= 25 ? 3.0f : (health <= 55 ? 1.8f : 1.0f);
        score += food_visible
            ? hunger * 16.0f * static_cast<float>(p.w + p.h - food_distance)
            : -2.0f * static_cast<float>(own_tail_distance);
        score += eats ? 180.0f * hunger : 0.0f;
        score += 2.0f * static_cast<float>(exits);
        score += move == previous_action ? 10.0f : 0.0f;
        score -= 0.5f * static_cast<float>(wall);
        score -= exits <= 1 ? 18.0f : 0.0f;
        score += winning_contest ? 12.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_HUNTER) {
        // Mirrors aggressive 30/32/41 play: visible heads are closed down,
        // especially when a length advantage makes the contest profitable.
        int target = nearest_prey <= p.w + p.h ? nearest_prey : nearest_enemy;
        if (target <= p.w + p.h) {
            score += 18.0f * static_cast<float>(p.w + p.h - target);
        }
        score += 6.0f * static_cast<float>(exits);
        score += winning_contest ? 260.0f : 0.0f;
        if (health <= 40 && food_visible) {
            score += 9.0f * static_cast<float>(p.w + p.h - food_distance);
            score += eats ? 90.0f : 0.0f;
        }
        score += move == previous_action ? 3.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_TERRITORIAL) {
        // Mirrors central controllers 38/46: maximize durable territory,
        // branching factor, and wall clearance; fight only when necessary.
        int reachable = blackout_observed_reachable_space(
            p, a, game, snake, head_x, head_y, nx, ny, scratch
        );
        score += 2.5f * static_cast<float>(reachable);
        score += 18.0f * static_cast<float>(exits);
        score += 9.0f * static_cast<float>(wall);
        int minimum_space = snake_length + 4;
        if (reachable < minimum_space) {
            score -= 45.0f * static_cast<float>(minimum_space - reachable);
        }
        if (health <= 50 && food_visible) {
            score += 7.0f * static_cast<float>(p.w + p.h - food_distance);
        }
        score -= winning_contest ? 35.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_EDGE_TRAPPER) {
        // Mirrors edge-heavy 28/34/43 play and adds purposeful pressure along
        // walls. A preferred wall distance of one produces lanes and cutoffs.
        score += 20.0f * static_cast<float>(4 - abs(wall - 1));
        score += 5.0f * static_cast<float>(exits);
        if (nearest_edge_enemy <= p.w + p.h) {
            score += 12.0f * static_cast<float>(p.w + p.h - nearest_edge_enemy);
        }
        if (food_visible) {
            score += (health <= 55 ? 7.0f : 2.0f) *
                     static_cast<float>(p.w + p.h - food_distance);
        }
        score += winning_contest ? 150.0f : 0.0f;
        score += exits == 1 ? 8.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_SURVIVOR) {
        // Mirrors evasive, high-turn-rate 15/17/21 play. It values escape
        // routes, separation, tail access, and deterministic score jitter.
        int reachable = blackout_observed_reachable_space(
            p, a, game, snake, head_x, head_y, nx, ny, scratch
        );
        score += 1.75f * static_cast<float>(reachable);
        score += 16.0f * static_cast<float>(exits);
        score += 3.0f * static_cast<float>(wall);
        if (nearest_danger <= p.w + p.h) {
            score += 8.0f * static_cast<float>(nearest_danger);
        } else if (nearest_enemy <= p.w + p.h) {
            score += 4.0f * static_cast<float>(nearest_enemy);
        }
        score -= 1.5f * static_cast<float>(own_tail_distance);
        score += previous_action >= 0 && move != previous_action ? 12.0f : 0.0f;
        unsigned int jitter = static_cast<unsigned int>(
            game * 73 + snake * 29 + turn * 17 + move * 11
        );
        score += static_cast<float>((jitter ^ (jitter >> 3)) & 7u);
        if (health <= 35 && food_visible) {
            score += 8.0f * static_cast<float>(p.w + p.h - food_distance);
        }
        score -= winning_contest ? 80.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_SNAKE25) {
        // Snake 25 in the July 25-30 logs is primarily a local-space player:
        // 79% of visible moves maximize immediate exits, 51% approach food,
        // and head contests are overwhelmingly taken against shorter snakes.
        // A mild turn preference reproduces its roughly 40/53 straight/turn
        // split without making deterministic loops dominate the policy.
        int reachable = blackout_observed_reachable_space(
            p, a, game, snake, head_x, head_y, nx, ny, scratch
        );
        score += 1.35f * static_cast<float>(reachable);
        score += 21.0f * static_cast<float>(exits);
        score += 1.5f * static_cast<float>(wall);
        int minimum_space = snake_length + 5;
        if (minimum_space < 12) {
            minimum_space = 12;
        }
        if (reachable < minimum_space) {
            score -= 38.0f * static_cast<float>(minimum_space - reachable);
        }
        if (food_visible) {
            float food_weight = health <= 30 ? 10.0f :
                                (health <= 60 ? 6.0f : 4.0f);
            score += food_weight *
                     static_cast<float>(p.w + p.h - food_distance);
            score += eats ? 65.0f * food_weight : 0.0f;
        }
        score += winning_contest ? 240.0f : 0.0f;
        score += previous_action >= 0 && move != previous_action ? 5.0f : 0.0f;
        score += 0.4f * static_cast<float>(own_tail_distance);
        score -= exits <= 1 ? 75.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_SNAKE25_INTERCEPTOR) {
        // Counter 1 combines the proven full-state safety core of `best` with
        // a selective interception of Snake 25's predicted exit-rich route.
        // It is deliberately low-food and territory-first, unlike Denier.
        int reachable = blackout_reachable_space(
            p, a, game, nx, ny, scratch
        );
        score += static_cast<float>(reachable);
        score += 4.0f * static_cast<float>(exits);
        score += 0.5f * static_cast<float>(wall);
        int minimum_space = snake_length + 2;
        if (minimum_space < 8) {
            minimum_space = 8;
        }
        if (reachable < minimum_space) {
            score -= 24.0f * static_cast<float>(minimum_space - reachable);
        }
        if (exits <= 1) {
            score -= 18.0f;
        }
        if (food_visible) {
            float food_weight = health <= 25 ? 9.0f :
                                (health <= 50 ? 5.0f :
                                 (health <= 75 ? 2.0f : 0.75f));
            score += food_weight *
                     static_cast<float>(p.w + p.h - food_distance);
            score += eats ? 8.0f * food_weight : 0.0f;
        } else {
            score += 0.25f *
                     static_cast<float>(p.w + p.h - own_tail_distance);
        }
        int predicted_prey_distance = p.w + p.h + 1;
        bool predicted_collision = false;
        for (int other = 0; other < p.num_snakes; ++other) {
            if (other == snake || action_masks[other] == 0) {
                continue;
            }
            int oi = snake_index(p, game, other);
            if (!a.alive[oi] || a.body_size[oi] <= 0) {
                continue;
            }
            int other_head = body_index(p, game, other, 0);
            int ox = a.body_x[other_head];
            int oy = a.body_y[other_head];
            if (!blackout_head_visible(head_x, head_y, ox, oy)) {
                continue;
            }
            int enemy_previous = blackout_previous_action(p, a, game, other);
            int predicted_x = ox;
            int predicted_y = oy;
            float predicted_score = -1.0e30f;
            for (int enemy_move = 0; enemy_move < 4; ++enemy_move) {
                if (((action_masks[other] >> enemy_move) & 1) == 0) {
                    continue;
                }
                int tx;
                int ty;
                new_position(p, ox, oy, enemy_move, &tx, &ty);
                int enemy_exits = blackout_open_neighbors(p, a, game, tx, ty);
                int enemy_food = blackout_visible_food_distance(
                    p, a, game, turn, ox, oy, tx, ty
                );
                float candidate = 21.0f * static_cast<float>(enemy_exits) +
                                  1.5f * static_cast<float>(
                                      blackout_wall_distance(p, tx, ty)
                                  );
                if (enemy_food <= p.w + p.h) {
                    candidate += 5.0f *
                                 static_cast<float>(p.w + p.h - enemy_food);
                    candidate += enemy_food == 0 ? 150.0f : 0.0f;
                }
                candidate += enemy_previous >= 0 && enemy_move != enemy_previous
                    ? 5.0f : 0.0f;
                if (candidate > predicted_score) {
                    predicted_score = candidate;
                    predicted_x = tx;
                    predicted_y = ty;
                }
            }
            int distance = abs(nx - predicted_x) + abs(ny - predicted_y);
            if (snake_length > a.length[oi]) {
                if (distance < predicted_prey_distance) {
                    predicted_prey_distance = distance;
                }
                if (nx == predicted_x && ny == predicted_y) {
                    predicted_collision = true;
                }
            }
        }
        if (predicted_prey_distance <= p.w + p.h) {
            score += 8.0f * static_cast<float>(
                p.w + p.h - predicted_prey_distance
            );
        } else if (nearest_danger <= p.w + p.h) {
            score += 2.0f * static_cast<float>(nearest_danger);
        }
        score += predicted_collision ? 360.0f : 0.0f;
        score += winning_contest ? 60.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_SNAKE25_DENIER) {
        // Counter 2 contests resources instead of mirroring the interceptor.
        // It takes visible food races aggressively, stays in broad territory,
        // then converts a length lead into controlled head pressure.
        int reachable = blackout_observed_reachable_space(
            p, a, game, snake, head_x, head_y, nx, ny, scratch
        );
        score += 1.7f * static_cast<float>(reachable);
        score += 17.0f * static_cast<float>(exits);
        score += 3.5f * static_cast<float>(wall);
        if (food_visible) {
            float deny_weight = health <= 35 ? 13.0f : 9.0f;
            score += deny_weight *
                     static_cast<float>(p.w + p.h - food_distance);
            score += eats ? 120.0f * deny_weight : 0.0f;
        }
        if (nearest_prey <= p.w + p.h) {
            score += 8.0f * static_cast<float>(p.w + p.h - nearest_prey);
            score += winning_contest ? 280.0f : 0.0f;
        } else if (nearest_danger <= p.w + p.h) {
            score += 5.0f * static_cast<float>(nearest_danger);
        }
        int minimum_space = snake_length + 7;
        if (minimum_space < 14) {
            minimum_space = 14;
        }
        if (reachable < minimum_space) {
            score -= 42.0f * static_cast<float>(minimum_space - reachable);
        }
        score -= exits <= 1 ? 110.0f : 0.0f;
    } else if (profile == BLACKOUT_HEURISTIC_SNAKE25_DUELIST) {
        // In the August 1-2 logs Snake 25 won 18/19 games that reached a pure
        // 1-v-1. Its late-game policy is more patient than the general clone:
        // it follows its own tail, stays central, turns frequently, keeps food
        // pressure, and only closes hard when a length lead makes a head
        // contest favorable. It also accepts one-exit tail corridors more
        // often than its multi-snake policy (16% versus 6%).
        int reachable = blackout_observed_reachable_space(
            p, a, game, snake, head_x, head_y, nx, ny, scratch
        );
        score += 1.6f * static_cast<float>(reachable);
        score += 12.0f * static_cast<float>(exits);
        score += 6.0f * static_cast<float>(wall);
        int minimum_space = snake_length + 3;
        if (minimum_space < 10) {
            minimum_space = 10;
        }
        if (reachable < minimum_space) {
            score -= 30.0f * static_cast<float>(minimum_space - reachable);
        }
        if (food_visible) {
            float food_weight = health <= 30 ? 11.0f :
                                (health <= 60 ? 7.0f : 4.5f);
            score += food_weight *
                     static_cast<float>(p.w + p.h - food_distance);
            score += eats ? 70.0f * food_weight : 0.0f;
        }
        if (nearest_prey <= p.w + p.h) {
            score += 10.0f * static_cast<float>(
                p.w + p.h - nearest_prey
            );
            score += winning_contest ? 320.0f : 0.0f;
        } else if (nearest_danger <= p.w + p.h) {
            score += 3.0f * static_cast<float>(nearest_danger);
        }
        score -= 4.0f * static_cast<float>(own_tail_distance);
        score += previous_action >= 0 && move != previous_action ? 12.0f : 0.0f;
        score -= exits <= 1 ? 15.0f : 0.0f;
    }

    if (losing_contest) {
        score -= 2000.0f;
    }
    if (hazard) {
        score -= health <= p.hazard_damage + 2 ? 2000.0f : 35.0f;
    }
    if (exits == 0) {
        score -= 3000.0f;
    }
    return score;
}

__device__ int blackout_heuristic_action(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int snake,
    int turn,
    int profile,
    const int* action_masks,
    unsigned char* scratch
) {
    int mask = action_masks[snake];
    if (mask == 0 || profile < 0 || profile >= BLACKOUT_HEURISTIC_COUNT) {
        return 0;
    }
    int tie_start = (game * 3 + snake + turn * 5 + profile * 7) & 3;
    int best_move = 0;
    int best_tie_rank = 5;
    float best_score = -1.0e30f;
    for (int move = 0; move < 4; ++move) {
        if (((mask >> move) & 1) == 0) {
            continue;
        }
        float score = blackout_profile_score(
            p, a, game, snake, turn, profile, move, action_masks, scratch
        );
        int tie_rank = (move - tie_start + 4) & 3;
        if (score > best_score || (score == best_score && tie_rank < best_tie_rank)) {
            best_score = score;
            best_move = move;
            best_tie_rank = tie_rank;
        }
    }
    return best_move;
}

__device__ float blackout_kill_reward(int before, int after, bool self_after) {
    if (after == 0 || after == before) {
        return 0.0f;
    }
    if (self_after) {
        return static_cast<float>(before - after) / 3.0f;
    }
    return -static_cast<float>(after) / 3.0f;
}

__device__ int obs_idx(int game, int channel, int x, int y) {
    return game * BLACKOUT_OBS_SIZE +
           channel * BLACKOUT_OBS_XY * BLACKOUT_OBS_XY +
           x * BLACKOUT_OBS_XY +
           y;
}

__device__ void write_obs_cell(
    float* obs,
    int game,
    int channel,
    int x,
    int y,
    float value
) {
    if (x < 0 || y < 0 || x >= BLACKOUT_OBS_XY || y >= BLACKOUT_OBS_XY) {
        return;
    }
    obs[obs_idx(game, channel, x, y)] = value;
}

__device__ bool blackout_visible(int obs_x, int obs_y) {
    return abs(obs_x - 14) + abs(obs_y - 14) <= BLACKOUT_VIEW_RADIUS;
}

__device__ void encode_blackout_obs(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int self_snake,
    int turn,
    float* obs,
    unsigned char* legal_mask,
    int out_game
) {
    for (int i = 0; i < BLACKOUT_OBS_SIZE; ++i) {
        obs[out_game * BLACKOUT_OBS_SIZE + i] = 0.0f;
    }

    int self_idx = snake_index(p, game, self_snake);
    int legal = legal_action_mask(p, a, game, self_snake);
    for (int move = 0; move < 4; ++move) {
        legal_mask[out_game * 4 + move] = ((legal >> move) & 1) ? 1 : 0;
    }

    if (!a.alive[self_idx] || a.body_size[self_idx] <= 0) {
        return;
    }

    int head_bi = body_index(p, game, self_snake, 0);
    int head_x = a.body_x[head_bi];
    int head_y = a.body_y[head_bi];
    int x_off = p.w - head_x - 1;
    int y_off = p.h - head_y - 1;

    for (int x = 0; x < BLACKOUT_OBS_XY; ++x) {
        for (int y = 0; y < BLACKOUT_OBS_XY; ++y) {
            int bx = x - x_off;
            int by = y - y_off;
            float board_value = (bx >= 0 && by >= 0 && bx < p.w && by < p.h) ? 1.0f : -1.0f;
                write_obs_cell(obs, out_game, 1, x, y, board_value);
            if (blackout_visible(x, y)) {
                write_obs_cell(obs, out_game, 8, x, y, 1.0f);
            }
        }
    }

    int food_count = a.food_count[game];
    for (int i = 0; i < food_count; ++i) {
        int fidx = food_index(p, game, i);
        int fx = a.food_x[fidx];
        int fy = a.food_y[fidx];
        int ox = fx + x_off;
        int oy = fy + y_off;
        bool visible = blackout_visible(ox, oy) || a.food_turn[fidx] == turn;
        if (visible) {
            write_obs_cell(obs, out_game, 0, ox, oy, 1.0f);
        }
        write_obs_cell(obs, out_game, BLACKOUT_PRIV_FOOD, ox, oy, 1.0f);
    }

    int body_size = a.body_size[self_idx];
    float body_counter = static_cast<float>(a.length[self_idx]);
    for (int i = 0; i < body_size; ++i) {
        int bi = body_index(p, game, self_snake, i);
        int ox = a.body_x[bi] + x_off;
        int oy = a.body_y[bi] + y_off;
        write_obs_cell(obs, out_game, 2, ox, oy, body_counter / 10.0f);
        body_counter -= 1.0f;
    }
    write_obs_cell(obs, out_game, 3, head_x + x_off, head_y + y_off, 1.0f);
    float health_value = static_cast<float>(a.health[self_idx]) / 100.0f;
    for (int x = 0; x < BLACKOUT_OBS_XY; ++x) {
        for (int y = 0; y < BLACKOUT_OBS_XY; ++y) {
            write_obs_cell(obs, out_game, 4, x, y, health_value);
        }
    }
    if (body_size > 0) {
        int tail_bi = body_index(p, game, self_snake, body_size - 1);
        write_obs_cell(
            obs,
            out_game,
            5,
            a.body_x[tail_bi] + x_off,
            a.body_y[tail_bi] + y_off,
            1.0f
        );
    }

    for (int s = 0; s < p.num_snakes; ++s) {
        if (s == self_snake) {
            continue;
        }
        int si = snake_index(p, game, s);
        if (!a.alive[si]) {
            continue;
        }
        int enemy_body_size = a.body_size[si];
        float enemy_health = static_cast<float>(a.health[si]) / 100.0f;
        float enemy_counter = static_cast<float>(a.length[si]);
        for (int i = 0; i < enemy_body_size; ++i) {
            int bi = body_index(p, game, s, i);
            int ox = a.body_x[bi] + x_off;
            int oy = a.body_y[bi] + y_off;
            if (blackout_visible(ox, oy)) {
                write_obs_cell(obs, out_game, 6, ox, oy, 1.0f);
            }
            write_obs_cell(
                obs, out_game, BLACKOUT_PRIV_ENEMY_BODY, ox, oy, enemy_counter / 10.0f
            );
            write_obs_cell(
                obs, out_game, BLACKOUT_PRIV_ENEMY_HEALTH, ox, oy, enemy_health
            );
            enemy_counter -= 1.0f;
        }
        if (enemy_body_size > 0) {
            int ehi = body_index(p, game, s, 0);
            int ox = a.body_x[ehi] + x_off;
            int oy = a.body_y[ehi] + y_off;
            if (blackout_visible(ox, oy)) {
                write_obs_cell(obs, out_game, 7, ox, oy, 1.0f);
            }
            write_obs_cell(obs, out_game, BLACKOUT_PRIV_ENEMY_HEAD, ox, oy, 1.0f);
            int eti = body_index(p, game, s, enemy_body_size - 1);
            write_obs_cell(
                obs,
                out_game,
                BLACKOUT_PRIV_ENEMY_TAIL,
                a.body_x[eti] + x_off,
                a.body_y[eti] + y_off,
                1.0f
            );
        }
    }
}

__device__ void encode_blackout_obs(
    const SimParams& p,
    const DeviceArrays& a,
    int game,
    int turn,
    float* obs,
    unsigned char* legal_mask
) {
    encode_blackout_obs(p, a, game, 0, turn, obs, legal_mask, game);
}

__global__ void blackout_vec_reset_kernel(BlackoutVecEnv env) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }
    unsigned long long rng_state = env.d_rng_states[game];
    initialize_game(env.p, env.init, env.a, game, &rng_state);
    env.d_rng_states[game] = rng_state;
    env.d_turns[game] = env.p.init_turns_played;
    env.d_done[game] = 0;
    env.d_rewards[game] = 0.0f;
    encode_blackout_obs(
        env.p,
        env.a,
        game,
        env.d_turns[game],
        env.d_obs,
        env.d_legal_mask
    );
}

__global__ void blackout_vec_seed_kernel(BlackoutVecEnv env, unsigned long long seed) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }
    unsigned long long rng_state =
        seed ^ (0xD1B54A32D192ED03ull * static_cast<unsigned long long>(game + 1));
    env.d_rng_states[game] = splitmix64_next(&rng_state);
}

__global__ void blackout_vec_prepare_best_actions_kernel(
    BlackoutVecEnv env,
    int* out_actions
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }

    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int sole_at_turn = -1;
    compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    for (int snake = 0; snake < env.p.num_snakes; ++snake) {
        int output_index = game * env.p.num_snakes + snake;
        out_actions[output_index] = 0;
        env.d_best_action_masks[output_index] = action_masks[snake];
    }
}

__global__ void blackout_vec_evaluate_best_actions_kernel(
    BlackoutVecEnv env,
    const unsigned char* selection_mask,
    int* out_actions
) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_slots = env.num_envs * env.p.num_snakes;
    if (output_index >= num_slots) {
        return;
    }
    if (selection_mask != nullptr && selection_mask[output_index] == 0) {
        return;
    }

    int game = output_index / env.p.num_snakes;
    int snake = output_index % env.p.num_snakes;
    out_actions[output_index] = blackout_best_action(
        env.p,
        env.a,
        game,
        snake,
        env.d_turns[game],
        env.d_best_action_masks + game * env.p.num_snakes,
        env.d_best_scratch + output_index * env.p.cells
    );
}

__global__ void blackout_vec_heuristic_actions_kernel(
    BlackoutVecEnv env,
    const int* profile_ids,
    int* out_actions
) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_slots = env.num_envs * env.p.num_snakes;
    if (output_index >= num_slots) {
        return;
    }
    int profile = profile_ids[output_index];
    out_actions[output_index] = 0;
    if (profile < 0 || profile >= BLACKOUT_HEURISTIC_COUNT) {
        return;
    }
    int game = output_index / env.p.num_snakes;
    int snake = output_index % env.p.num_snakes;
    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int sole_at_turn = -1;
    compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    out_actions[output_index] = blackout_heuristic_action(
        env.p,
        env.a,
        game,
        snake,
        env.d_turns[game],
        profile,
        action_masks,
        env.d_best_scratch + output_index * env.p.cells
    );
}

__global__ void blackout_vec_step_kernel(BlackoutVecEnv env) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }

    unsigned long long rng_state = env.d_rng_states[game];
    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int sole_at_turn = -1;
    int at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    if (at_turn <= 1) {
        env.d_done[game] = 1;
        env.d_rewards[game] = 0.0f;
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        encode_blackout_obs(env.p, env.a, game, env.d_turns[game], env.d_obs, env.d_legal_mask);
        env.d_rng_states[game] = rng_state;
        return;
    }

    int actions[HISSS_CUDA_MAX_SNAKES];
    int requested = env.d_actions[game];
    // The learning agent's requested move is the transition that must actually
    // happen. Invalid-but-in-range moves therefore result in the corresponding
    // collision/starvation instead of being silently replaced.
    actions[0] = requested;

    for (int s = 1; s < env.p.num_snakes; ++s) {
        actions[s] = choose_action_from_mask(&rng_state, action_masks[s]);
    }

    int prev_turn = env.d_turns[game];
    int next_turn = step_game(env.p, env.a, game, prev_turn, actions, &rng_state);
    env.d_turns[game] = next_turn;

    int before = at_turn;
    at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    bool self_after = action_masks[0] != 0;
    float reward = blackout_kill_reward(before, at_turn, self_after);
    bool done = at_turn <= 1 || !self_after;

    env.d_done[game] = done ? 1 : 0;
    env.d_rewards[game] = reward;
    env.d_rng_states[game] = rng_state;
    if (done) {
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        env.d_rng_states[game] = rng_state;
        encode_blackout_obs(env.p, env.a, game, env.d_turns[game], env.d_obs, env.d_legal_mask);
    } else {
        encode_blackout_obs(env.p, env.a, game, next_turn, env.d_obs, env.d_legal_mask);
    }
}

// Observation encoding dominates the vector environment when four seats are
// requested. The original implementation assigned all 30,276 output values of
// a game to one CUDA thread. Keep sparse body/food writes together per seat,
// but parallelize the three dense planes over their 841 spatial cells.
template <typename ObsT>
__device__ void write_obs_cell_t(
    ObsT* obs,
    int game,
    int channel,
    int x,
    int y,
    float value
) {
    if (x < 0 || y < 0 || x >= BLACKOUT_OBS_XY || y >= BLACKOUT_OBS_XY) {
        return;
    }
    obs[obs_idx(game, channel, x, y)] = static_cast<ObsT>(value);
}

template <typename ObsT>
__global__ void blackout_encode_dense_all_kernel(
    BlackoutVecEnv env,
    ObsT* out_obs
) {
    int output_cell = blockIdx.x * blockDim.x + threadIdx.x;
    int num_outputs = env.num_envs * env.p.num_snakes;
    int total_cells = num_outputs * BLACKOUT_OBS_PLANE;
    if (output_cell >= total_cells) {
        return;
    }

    int out_game = output_cell / BLACKOUT_OBS_PLANE;
    int spatial = output_cell - out_game * BLACKOUT_OBS_PLANE;
    int game = out_game / env.p.num_snakes;
    int self_snake = out_game - game * env.p.num_snakes;
    int self_idx = snake_index(env.p, game, self_snake);
    if (!env.a.alive[self_idx] || env.a.body_size[self_idx] <= 0) {
        return;
    }

    int head = body_index(env.p, game, self_snake, 0);
    int head_x = env.a.body_x[head];
    int head_y = env.a.body_y[head];
    int x_off = env.p.w - head_x - 1;
    int y_off = env.p.h - head_y - 1;
    int x = spatial / BLACKOUT_OBS_XY;
    int y = spatial - x * BLACKOUT_OBS_XY;
    int board_x = x - x_off;
    int board_y = y - y_off;
    int base = out_game * BLACKOUT_OBS_SIZE + spatial;

    out_obs[base + BLACKOUT_OBS_PLANE] = static_cast<ObsT>(
        in_bounds(env.p, board_x, board_y) ? 1.0f : -1.0f
    );
    out_obs[base + 4 * BLACKOUT_OBS_PLANE] = static_cast<ObsT>(
        static_cast<float>(env.a.health[self_idx]) / 100.0f
    );
    if (blackout_visible(x, y)) {
        out_obs[base + 8 * BLACKOUT_OBS_PLANE] = static_cast<ObsT>(1.0f);
    }
}

template <typename ObsT>
__global__ void blackout_encode_sparse_all_kernel(
    BlackoutVecEnv env,
    ObsT* out_obs,
    unsigned char* out_legal_mask
) {
    int out_game = blockIdx.x * blockDim.x + threadIdx.x;
    int num_outputs = env.num_envs * env.p.num_snakes;
    if (out_game >= num_outputs) {
        return;
    }

    int game = out_game / env.p.num_snakes;
    int self_snake = out_game - game * env.p.num_snakes;
    int legal = legal_action_mask(env.p, env.a, game, self_snake);
    for (int move = 0; move < 4; ++move) {
        out_legal_mask[out_game * 4 + move] = ((legal >> move) & 1) ? 1 : 0;
    }

    int self_idx = snake_index(env.p, game, self_snake);
    if (!env.a.alive[self_idx] || env.a.body_size[self_idx] <= 0) {
        return;
    }
    int head = body_index(env.p, game, self_snake, 0);
    int head_x = env.a.body_x[head];
    int head_y = env.a.body_y[head];
    int x_off = env.p.w - head_x - 1;
    int y_off = env.p.h - head_y - 1;

    int food_count = env.a.food_count[game];
    for (int food = 0; food < food_count; ++food) {
        int fi = food_index(env.p, game, food);
        int ox = env.a.food_x[fi] + x_off;
        int oy = env.a.food_y[fi] + y_off;
        if (blackout_visible(ox, oy) || env.a.food_turn[fi] == env.d_turns[game]) {
            write_obs_cell_t(out_obs, out_game, 0, ox, oy, 1.0f);
        }
        write_obs_cell_t(out_obs, out_game, BLACKOUT_PRIV_FOOD, ox, oy, 1.0f);
    }

    int body_size = env.a.body_size[self_idx];
    float body_counter = static_cast<float>(env.a.length[self_idx]);
    for (int body_pos = 0; body_pos < body_size; ++body_pos) {
        int body = body_index(env.p, game, self_snake, body_pos);
        write_obs_cell_t(
            out_obs,
            out_game,
            2,
            env.a.body_x[body] + x_off,
            env.a.body_y[body] + y_off,
            body_counter / 10.0f
        );
        body_counter -= 1.0f;
    }
    write_obs_cell_t(out_obs, out_game, 3, 14, 14, 1.0f);
    int tail = body_index(env.p, game, self_snake, body_size - 1);
    write_obs_cell_t(
        out_obs,
        out_game,
        5,
        env.a.body_x[tail] + x_off,
        env.a.body_y[tail] + y_off,
        1.0f
    );

    for (int snake = 0; snake < env.p.num_snakes; ++snake) {
        if (snake == self_snake) {
            continue;
        }
        int si = snake_index(env.p, game, snake);
        if (!env.a.alive[si]) {
            continue;
        }
        int enemy_size = env.a.body_size[si];
        float enemy_health = static_cast<float>(env.a.health[si]) / 100.0f;
        float enemy_counter = static_cast<float>(env.a.length[si]);
        for (int body_pos = 0; body_pos < enemy_size; ++body_pos) {
            int body = body_index(env.p, game, snake, body_pos);
            int ox = env.a.body_x[body] + x_off;
            int oy = env.a.body_y[body] + y_off;
            if (blackout_visible(ox, oy)) {
                write_obs_cell_t(out_obs, out_game, 6, ox, oy, 1.0f);
            }
            write_obs_cell_t(
                out_obs,
                out_game,
                BLACKOUT_PRIV_ENEMY_BODY,
                ox,
                oy,
                enemy_counter / 10.0f
            );
            write_obs_cell_t(
                out_obs,
                out_game,
                BLACKOUT_PRIV_ENEMY_HEALTH,
                ox,
                oy,
                enemy_health
            );
            enemy_counter -= 1.0f;
        }
        if (enemy_size > 0) {
            int enemy_head = body_index(env.p, game, snake, 0);
            int ox = env.a.body_x[enemy_head] + x_off;
            int oy = env.a.body_y[enemy_head] + y_off;
            if (blackout_visible(ox, oy)) {
                write_obs_cell_t(out_obs, out_game, 7, ox, oy, 1.0f);
            }
            write_obs_cell_t(
                out_obs, out_game, BLACKOUT_PRIV_ENEMY_HEAD, ox, oy, 1.0f
            );
            int enemy_tail = body_index(env.p, game, snake, enemy_size - 1);
            write_obs_cell_t(
                out_obs,
                out_game,
                BLACKOUT_PRIV_ENEMY_TAIL,
                env.a.body_x[enemy_tail] + x_off,
                env.a.body_y[enemy_tail] + y_off,
                1.0f
            );
        }
    }
}

template <typename ObsT>
cudaError_t launch_blackout_encode_all(
    const BlackoutVecEnv& env,
    ObsT* out_obs,
    unsigned char* out_legal_mask,
    cudaStream_t stream = nullptr
) {
    size_t obs_count = static_cast<size_t>(env.num_envs) *
                       static_cast<size_t>(env.p.num_snakes) *
                       static_cast<size_t>(BLACKOUT_OBS_SIZE);
    cudaError_t err = cudaMemsetAsync(
        out_obs,
        0,
        obs_count * sizeof(ObsT),
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }
    constexpr int dense_block_size = HISSS_CUDA_DENSE_BLOCK_SIZE;
    int num_outputs = env.num_envs * env.p.num_snakes;
    int dense_items = num_outputs * BLACKOUT_OBS_PLANE;
    int dense_grid = (dense_items + dense_block_size - 1) / dense_block_size;
    blackout_encode_dense_all_kernel<ObsT><<<
        dense_grid, dense_block_size, 0, stream
    >>>(
        env,
        out_obs
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    int sparse_grid =
        (num_outputs + HISSS_CUDA_GAME_BLOCK_SIZE - 1) /
        HISSS_CUDA_GAME_BLOCK_SIZE;
    blackout_encode_sparse_all_kernel<ObsT><<<
        sparse_grid, HISSS_CUDA_GAME_BLOCK_SIZE, 0, stream
    >>>(
        env,
        out_obs,
        out_legal_mask
    );
    return cudaGetLastError();
}

__global__ void blackout_vec_reset_all_kernel(BlackoutVecEnv env) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }
    unsigned long long rng_state = env.d_rng_states[game];
    initialize_game(env.p, env.init, env.a, game, &rng_state);
    env.d_turns[game] = env.p.init_turns_played;
    env.d_done[game] = 0;
    env.d_rewards[game] = 0.0f;
    env.d_rng_states[game] = rng_state;
}

__global__ void blackout_vec_step_all_kernel(
    BlackoutVecEnv env,
    const int* all_actions,
    float* out_rewards,
    unsigned char* out_done
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }

    unsigned long long rng_state = env.d_rng_states[game];
    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int sole_at_turn = -1;
    int at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    if (at_turn <= 1) {
        env.d_done[game] = 1;
        env.d_rewards[game] = 0.0f;
        out_done[game] = 1;
        out_rewards[game] = 0.0f;
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        env.d_rng_states[game] = rng_state;
        return;
    }

    int actions[HISSS_CUDA_MAX_SNAKES];
    for (int s = 0; s < env.p.num_snakes; ++s) {
        int requested = all_actions[game * env.p.num_snakes + s];
        if (s == 0) {
            actions[s] = requested;
            continue;
        }
        int mask = action_masks[s];
        if (requested < 0 || requested >= 4 || ((mask >> requested) & 1) == 0) {
            actions[s] = choose_action_from_mask(&rng_state, mask);
        } else {
            actions[s] = requested;
        }
    }

    int prev_turn = env.d_turns[game];
    int next_turn = step_game(env.p, env.a, game, prev_turn, actions, &rng_state);
    env.d_turns[game] = next_turn;

    int before = at_turn;
    at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    bool self_after = action_masks[0] != 0;
    float reward = blackout_kill_reward(before, at_turn, self_after);
    bool done = at_turn <= 1 || !self_after;

    env.d_done[game] = done ? 1 : 0;
    env.d_rewards[game] = reward;
    out_done[game] = done ? 1 : 0;
    out_rewards[game] = reward;
    env.d_rng_states[game] = rng_state;
    if (done) {
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        env.d_rng_states[game] = rng_state;
    }
}

__device__ void write_eval_state(
    const BlackoutVecEnv& env,
    int game,
    const int* action_masks,
    int winner,
    int turns,
    unsigned char* out_done,
    int* out_winner,
    unsigned char* out_alive,
    int* out_turns
) {
    out_done[game] = 1;
    out_winner[game] = winner;
    out_turns[game] = turns;
    for (int snake = 0; snake < env.p.num_snakes; ++snake) {
        out_alive[game * env.p.num_snakes + snake] =
            action_masks[snake] != 0 ? 1 : 0;
    }
}

__global__ void blackout_vec_step_all_eval_kernel(
    BlackoutVecEnv env,
    const int* all_actions,
    int max_turns,
    unsigned char* out_done,
    int* out_winner,
    unsigned char* out_alive,
    int* out_turns
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= env.num_envs) {
        return;
    }

    unsigned long long rng_state = env.d_rng_states[game];
    int action_masks[HISSS_CUDA_MAX_SNAKES];
    int sole_at_turn = -1;
    int at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);

    // Normally the previous eval step has already reset terminal games. Keep
    // this guard so the API is also safe immediately after arbitrary calls.
    if (at_turn <= 1) {
        int winner = winner_from_state(env.p, env.a, game, at_turn, sole_at_turn);
        write_eval_state(
            env,
            game,
            action_masks,
            winner,
            env.d_turns[game],
            out_done,
            out_winner,
            out_alive,
            out_turns
        );
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        env.d_rng_states[game] = rng_state;
        return;
    }

    int actions[HISSS_CUDA_MAX_SNAKES];
    for (int snake = 0; snake < env.p.num_snakes; ++snake) {
        int requested = all_actions[game * env.p.num_snakes + snake];
        actions[snake] = requested >= 0 && requested < 4
            ? requested
            : choose_action_from_mask(&rng_state, action_masks[snake]);
    }

    int next_turn = step_game(
        env.p,
        env.a,
        game,
        env.d_turns[game],
        actions,
        &rng_state
    );
    env.d_turns[game] = next_turn;
    at_turn = compute_at_turn(env.p, env.a, game, action_masks, &sole_at_turn);
    bool terminal = at_turn <= 1;
    bool truncated = next_turn >= max_turns;
    bool done = terminal || truncated;

    out_done[game] = done ? 1 : 0;
    out_winner[game] = terminal
        ? winner_from_state(env.p, env.a, game, at_turn, sole_at_turn)
        : -1;
    out_turns[game] = next_turn;
    for (int snake = 0; snake < env.p.num_snakes; ++snake) {
        out_alive[game * env.p.num_snakes + snake] =
            action_masks[snake] != 0 ? 1 : 0;
    }

    env.d_done[game] = done ? 1 : 0;
    env.d_rewards[game] = 0.0f;
    env.d_rng_states[game] = rng_state;
    if (done) {
        initialize_game(env.p, env.init, env.a, game, &rng_state);
        env.d_turns[game] = env.p.init_turns_played;
        env.d_rng_states[game] = rng_state;
    }
}

bool blackout_alloc_common(BlackoutVecEnv* env) {
    size_t games_snakes = static_cast<size_t>(env->num_envs) * BLACKOUT_SNAKES;
    size_t body_entries = games_snakes * static_cast<size_t>(env->p.max_body_len);
    size_t board_entries = games_snakes * static_cast<size_t>(env->p.cells);
    size_t food_entries = static_cast<size_t>(env->num_envs) * static_cast<size_t>(env->p.max_food);
    size_t game_cells = static_cast<size_t>(env->num_envs) * static_cast<size_t>(env->p.cells);
    bool ok = true;
    ok = ok && device_alloc(&env->a.body_x, body_entries, "vec_body_x");
    ok = ok && device_alloc(&env->a.body_y, body_entries, "vec_body_y");
    ok = ok && device_alloc(&env->a.body_size, games_snakes, "vec_body_size");
    ok = ok && device_alloc(&env->a.board, board_entries, "vec_board");
    ok = ok && device_alloc(&env->a.food_x, food_entries, "vec_food_x");
    ok = ok && device_alloc(&env->a.food_y, food_entries, "vec_food_y");
    ok = ok && device_alloc(&env->a.food_turn, food_entries, "vec_food_turn");
    ok = ok && device_alloc(&env->a.food_count, env->num_envs, "vec_food_count");
    ok = ok && device_alloc(&env->a.scratch, game_cells, "vec_scratch");
    ok = ok && device_alloc(&env->a.hazards, game_cells, "vec_hazards");
    ok = ok && device_alloc(&env->a.alive, games_snakes, "vec_alive");
    ok = ok && device_alloc(&env->a.health, games_snakes, "vec_health");
    ok = ok && device_alloc(&env->a.length, games_snakes, "vec_length");
    ok = ok && device_alloc(&env->a.max_health, games_snakes, "vec_max_health");
    ok = ok && device_alloc(&env->a.death_cause, games_snakes, "vec_death_cause");
    ok = ok && device_alloc(&env->a.death_turn, games_snakes, "vec_death_turn");
    ok = ok && device_alloc(&env->a.killer_id, games_snakes, "vec_killer_id");
    ok = ok && device_alloc(&env->a.out_turns_played, env->num_envs, "vec_out_turns");
    ok = ok && device_alloc(&env->a.out_terminal, env->num_envs, "vec_out_terminal");
    ok = ok && device_alloc(&env->a.out_winner, env->num_envs, "vec_out_winner");
    ok = ok && device_alloc(&env->d_turns, env->num_envs, "vec_turns");
    ok = ok && device_alloc(&env->d_rng_states, env->num_envs, "vec_rng");
    ok = ok && device_alloc(&env->d_actions, env->num_envs, "vec_actions");
    ok = ok && device_alloc(&env->d_obs, static_cast<size_t>(env->num_envs) * BLACKOUT_OBS_SIZE, "vec_obs");
    ok = ok && device_alloc(&env->d_rewards, env->num_envs, "vec_rewards");
    ok = ok && device_alloc(&env->d_done, env->num_envs, "vec_done");
    ok = ok && device_alloc(&env->d_legal_mask, static_cast<size_t>(env->num_envs) * 4u, "vec_legal");
    ok = ok && device_alloc(&env->d_best_action_masks, games_snakes, "vec_best_action_masks");
    ok = ok && device_alloc(&env->d_best_scratch, games_snakes * static_cast<size_t>(env->p.cells), "vec_best_scratch");
    return ok;
}

void blackout_free(BlackoutVecEnv* env) {
    if (env == nullptr) {
        return;
    }
    free_init_arrays(env->init);
    free_device_arrays(env->a);
    cudaFree(env->d_turns);
    cudaFree(env->d_rng_states);
    cudaFree(env->d_actions);
    cudaFree(env->d_obs);
    cudaFree(env->d_rewards);
    cudaFree(env->d_done);
    cudaFree(env->d_legal_mask);
    cudaFree(env->d_best_action_masks);
    cudaFree(env->d_best_scratch);
    delete env;
}

bool copy_blackout_outputs(
    BlackoutVecEnv* env,
    float* out_obs,
    float* out_rewards,
    bool* out_done,
    bool* out_legal_mask
) {
    bool ok = true;
    cudaError_t err = cudaMemcpy(
        out_obs,
        env->d_obs,
        static_cast<size_t>(env->num_envs) * BLACKOUT_OBS_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess) { set_cuda_error("copy vec obs", err); ok = false; }
    if (out_rewards != nullptr) {
        err = cudaMemcpy(out_rewards, env->d_rewards, env->num_envs * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { set_cuda_error("copy vec rewards", err); ok = false; }
    }
    if (out_done != nullptr) {
        err = cudaMemcpy(out_done, env->d_done, env->num_envs * sizeof(bool), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { set_cuda_error("copy vec done", err); ok = false; }
    }
    err = cudaMemcpy(
        out_legal_mask,
        env->d_legal_mask,
        static_cast<size_t>(env->num_envs) * 4u * sizeof(bool),
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess) { set_cuda_error("copy vec legal", err); ok = false; }
    return ok;
}

}  // namespace

void* hisss_cuda_blackout_vec_create(
    int num_envs,
    unsigned long long seed,
    int duel_probability_ppm
) {
    g_hisss_cuda_error.clear();
    if (num_envs <= 0) {
        set_last_error("num_envs must be positive");
        return nullptr;
    }
    if (duel_probability_ppm < 0 || duel_probability_ppm > 1000000) {
        set_last_error("duel_probability_ppm must be in [0, 1000000]");
        return nullptr;
    }

    auto* env = new BlackoutVecEnv();
    env->num_envs = num_envs;
    env->p.w = BLACKOUT_W;
    env->p.h = BLACKOUT_H;
    env->p.cells = BLACKOUT_W * BLACKOUT_H;
    env->p.num_snakes = BLACKOUT_SNAKES;
    env->p.min_food = 1;
    env->p.food_spawn_chance = 15;
    env->p.init_turns_played = 0;
    env->p.spawn_snakes_randomly = true;
    env->p.max_init_body_length = 0;
    env->p.num_init_food = -1;
    env->p.wrapped = false;
    env->p.royale = false;
    env->p.shrink_n_turns = 25;
    env->p.hazard_damage = 14;
    env->p.num_games = num_envs;
    env->p.max_turns = 0;
    env->p.max_body_len = BLACKOUT_W * BLACKOUT_H + 2;
    env->p.max_food = BLACKOUT_W * BLACKOUT_H;
    env->p.duel_probability_ppm = duel_probability_ppm;

    bool alive[BLACKOUT_SNAKES] = {true, true, true, true};
    int health[BLACKOUT_SNAKES] = {100, 100, 100, 100};
    int length[BLACKOUT_SNAKES] = {3, 3, 3, 3};
    int max_health[BLACKOUT_SNAKES] = {100, 100, 100, 100};
    bool hazards[BLACKOUT_W * BLACKOUT_H] = {false};

    bool ok = true;
    ok = ok && device_alloc_copy(&env->init.snake_alive, alive, BLACKOUT_SNAKES, "vec_init_alive");
    ok = ok && device_alloc_copy(&env->init.snake_health, health, BLACKOUT_SNAKES, "vec_init_health");
    ok = ok && device_alloc_copy(&env->init.snake_len, length, BLACKOUT_SNAKES, "vec_init_len");
    ok = ok && device_alloc_copy(&env->init.max_health, max_health, BLACKOUT_SNAKES, "vec_init_max_health");
    ok = ok && device_alloc_copy(&env->init.init_hazards, hazards, BLACKOUT_W * BLACKOUT_H, "vec_init_hazards");
    ok = ok && blackout_alloc_common(env);
    if (!ok) {
        blackout_free(env);
        return nullptr;
    }

    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (num_envs + block_size - 1) / block_size;
    blackout_vec_seed_kernel<<<grid_size, block_size>>>(*env, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_seed launch", err);
        blackout_free(env);
        return nullptr;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_seed execution", err);
        blackout_free(env);
        return nullptr;
    }
    return env;
}

void hisss_cuda_blackout_vec_close(void* env) {
    blackout_free(reinterpret_cast<BlackoutVecEnv*>(env));
}

int hisss_cuda_blackout_vec_reset(void* env_p, float* out_obs, bool* out_legal_mask) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    if (env == nullptr || out_obs == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec reset arguments");
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_reset_kernel<<<grid_size, block_size>>>(*env);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset launch", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset execution", err);
        return 1;
    }
    return copy_blackout_outputs(env, out_obs, nullptr, nullptr, out_legal_mask) ? 0 : 1;
}

int hisss_cuda_blackout_vec_step(
    void* env_p,
    const int* actions,
    float* out_obs,
    float* out_rewards,
    bool* out_done,
    bool* out_legal_mask
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    if (env == nullptr || actions == nullptr || out_obs == nullptr ||
        out_rewards == nullptr || out_done == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec step arguments");
        return 1;
    }
    cudaError_t err = cudaMemcpy(
        env->d_actions,
        actions,
        env->num_envs * sizeof(int),
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec actions", err);
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_step_kernel<<<grid_size, block_size>>>(*env);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step launch", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step execution", err);
        return 1;
    }
    return copy_blackout_outputs(env, out_obs, out_rewards, out_done, out_legal_mask) ? 0 : 1;
}

int hisss_cuda_blackout_vec_reset_device(
    void* env_p,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* out_obs = reinterpret_cast<float*>(out_obs_device_ptr);
    auto* out_legal_mask = reinterpret_cast<unsigned char*>(out_legal_mask_device_ptr);
    if (env == nullptr || out_obs == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec reset_device arguments");
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_reset_kernel<<<grid_size, block_size>>>(*env);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset_device launch", err);
        return 1;
    }
    err = cudaMemcpy(
        out_obs,
        env->d_obs,
        static_cast<size_t>(env->num_envs) * BLACKOUT_OBS_SIZE * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec obs device", err);
        return 1;
    }
    err = cudaMemcpy(
        out_legal_mask,
        env->d_legal_mask,
        static_cast<size_t>(env->num_envs) * 4u * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec legal device", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_reset_all_device(
    void* env_p,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr
) {
    int result = hisss_cuda_blackout_vec_reset_all_device_async(
        env_p,
        out_obs_device_ptr,
        out_legal_mask_device_ptr,
        false,
        0
    );
    if (result != 0) {
        return result;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset_all_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_reset_all_device_async(
    void* env_p,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    bool out_obs_fp16,
    uintptr_t stream_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* out_obs = reinterpret_cast<void*>(out_obs_device_ptr);
    auto* out_legal_mask = reinterpret_cast<unsigned char*>(out_legal_mask_device_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (env == nullptr || out_obs == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec reset_all_device arguments");
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_reset_all_kernel<<<grid_size, block_size, 0, stream>>>(*env);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset_all_device launch", err);
        return 1;
    }
    if (out_obs_fp16) {
        err = launch_blackout_encode_all(
            *env, reinterpret_cast<__half*>(out_obs), out_legal_mask, stream
        );
    } else {
        err = launch_blackout_encode_all(
            *env, reinterpret_cast<float*>(out_obs), out_legal_mask, stream
        );
    }
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_reset_all_device encode", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_step_device(
    void* env_p,
    uintptr_t actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* actions = reinterpret_cast<int*>(actions_device_ptr);
    auto* out_obs = reinterpret_cast<float*>(out_obs_device_ptr);
    auto* out_rewards = reinterpret_cast<float*>(out_rewards_device_ptr);
    auto* out_done = reinterpret_cast<unsigned char*>(out_done_device_ptr);
    auto* out_legal_mask = reinterpret_cast<unsigned char*>(out_legal_mask_device_ptr);
    if (env == nullptr || actions == nullptr || out_obs == nullptr ||
        out_rewards == nullptr || out_done == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec step_device arguments");
        return 1;
    }
    cudaError_t err = cudaMemcpy(
        env->d_actions,
        actions,
        env->num_envs * sizeof(int),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec actions device", err);
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_step_kernel<<<grid_size, block_size>>>(*env);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_device launch", err);
        return 1;
    }
    err = cudaMemcpy(
        out_obs,
        env->d_obs,
        static_cast<size_t>(env->num_envs) * BLACKOUT_OBS_SIZE * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec obs device", err);
        return 1;
    }
    err = cudaMemcpy(
        out_rewards,
        env->d_rewards,
        env->num_envs * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec rewards device", err);
        return 1;
    }
    err = cudaMemcpy(
        out_done,
        env->d_done,
        env->num_envs * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec done device", err);
        return 1;
    }
    err = cudaMemcpy(
        out_legal_mask,
        env->d_legal_mask,
        static_cast<size_t>(env->num_envs) * 4u * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        set_cuda_error("copy vec legal device", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_step_all_device_async(
    void* env_p,
    uintptr_t all_actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    bool out_obs_fp16,
    uintptr_t stream_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* all_actions = reinterpret_cast<int*>(all_actions_device_ptr);
    auto* out_obs = reinterpret_cast<void*>(out_obs_device_ptr);
    auto* out_rewards = reinterpret_cast<float*>(out_rewards_device_ptr);
    auto* out_done = reinterpret_cast<unsigned char*>(out_done_device_ptr);
    auto* out_legal_mask = reinterpret_cast<unsigned char*>(out_legal_mask_device_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (env == nullptr || all_actions == nullptr || out_obs == nullptr ||
        out_rewards == nullptr || out_done == nullptr || out_legal_mask == nullptr) {
        set_last_error("invalid blackout vec step_all_device arguments");
        return 1;
    }
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_step_all_kernel<<<grid_size, block_size, 0, stream>>>(
        *env,
        all_actions,
        out_rewards,
        out_done
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_device launch", err);
        return 1;
    }
    if (out_obs_fp16) {
        err = launch_blackout_encode_all(
            *env, reinterpret_cast<__half*>(out_obs), out_legal_mask, stream
        );
    } else {
        err = launch_blackout_encode_all(
            *env, reinterpret_cast<float*>(out_obs), out_legal_mask, stream
        );
    }
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_device encode", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_step_all_device(
    void* env_p,
    uintptr_t all_actions_device_ptr,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_rewards_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr
) {
    int result = hisss_cuda_blackout_vec_step_all_device_async(
        env_p,
        all_actions_device_ptr,
        out_obs_device_ptr,
        out_rewards_device_ptr,
        out_done_device_ptr,
        out_legal_mask_device_ptr,
        false,
        0
    );
    if (result != 0) {
        return result;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_step_all_eval_device(
    void* env_p,
    uintptr_t all_actions_device_ptr,
    int max_turns,
    uintptr_t out_obs_device_ptr,
    uintptr_t out_done_device_ptr,
    uintptr_t out_legal_mask_device_ptr,
    uintptr_t out_winner_device_ptr,
    uintptr_t out_alive_device_ptr,
    uintptr_t out_turns_device_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* all_actions = reinterpret_cast<int*>(all_actions_device_ptr);
    auto* out_obs = reinterpret_cast<float*>(out_obs_device_ptr);
    auto* out_done = reinterpret_cast<unsigned char*>(out_done_device_ptr);
    auto* out_legal_mask = reinterpret_cast<unsigned char*>(out_legal_mask_device_ptr);
    auto* out_winner = reinterpret_cast<int*>(out_winner_device_ptr);
    auto* out_alive = reinterpret_cast<unsigned char*>(out_alive_device_ptr);
    auto* out_turns = reinterpret_cast<int*>(out_turns_device_ptr);
    if (env == nullptr || all_actions == nullptr || max_turns <= 0 ||
        out_obs == nullptr || out_done == nullptr || out_legal_mask == nullptr ||
        out_winner == nullptr || out_alive == nullptr || out_turns == nullptr) {
        set_last_error("invalid blackout vec step_all_eval_device arguments");
        return 1;
    }

    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_step_all_eval_kernel<<<grid_size, block_size>>>(
        *env,
        all_actions,
        max_turns,
        out_done,
        out_winner,
        out_alive,
        out_turns
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_eval_device launch", err);
        return 1;
    }
    err = launch_blackout_encode_all(*env, out_obs, out_legal_mask);
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_eval_device encode", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_step_all_eval_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_best_actions_device_async(
    void* env_p,
    uintptr_t selection_mask_device_ptr,
    uintptr_t out_actions_device_ptr,
    uintptr_t stream_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* selection_mask = reinterpret_cast<unsigned char*>(selection_mask_device_ptr);
    auto* out_actions = reinterpret_cast<int*>(out_actions_device_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (env == nullptr || out_actions == nullptr) {
        set_last_error("invalid blackout vec best_actions_device arguments");
        return 1;
    }

    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (env->num_envs + block_size - 1) / block_size;
    blackout_vec_prepare_best_actions_kernel<<<grid_size, block_size, 0, stream>>>(
        *env,
        out_actions
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_prepare_best_actions_device launch", err);
        return 1;
    }

    int num_slots = env->num_envs * env->p.num_snakes;
    int evaluate_grid_size = (num_slots + block_size - 1) / block_size;
    blackout_vec_evaluate_best_actions_kernel<<<
        evaluate_grid_size, block_size, 0, stream
    >>>(
        *env,
        selection_mask,
        out_actions
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_evaluate_best_actions_device launch", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_best_actions_device(
    void* env_p,
    uintptr_t selection_mask_device_ptr,
    uintptr_t out_actions_device_ptr
) {
    int result = hisss_cuda_blackout_vec_best_actions_device_async(
        env_p,
        selection_mask_device_ptr,
        out_actions_device_ptr,
        0
    );
    if (result != 0) {
        return result;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_best_actions_device execution", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_heuristic_actions_device_async(
    void* env_p,
    uintptr_t profile_ids_device_ptr,
    uintptr_t out_actions_device_ptr,
    uintptr_t stream_ptr
) {
    g_hisss_cuda_error.clear();
    auto* env = reinterpret_cast<BlackoutVecEnv*>(env_p);
    auto* profile_ids = reinterpret_cast<int*>(profile_ids_device_ptr);
    auto* out_actions = reinterpret_cast<int*>(out_actions_device_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (env == nullptr || profile_ids == nullptr || out_actions == nullptr) {
        set_last_error("invalid blackout vec heuristic_actions_device arguments");
        return 1;
    }

    int num_slots = env->num_envs * env->p.num_snakes;
    int block_size = HISSS_CUDA_GAME_BLOCK_SIZE;
    int grid_size = (num_slots + block_size - 1) / block_size;
    blackout_vec_heuristic_actions_kernel<<<grid_size, block_size, 0, stream>>>(
        *env,
        profile_ids,
        out_actions
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_heuristic_actions_device launch", err);
        return 1;
    }
    return 0;
}

int hisss_cuda_blackout_vec_heuristic_actions_device(
    void* env_p,
    uintptr_t profile_ids_device_ptr,
    uintptr_t out_actions_device_ptr
) {
    int result = hisss_cuda_blackout_vec_heuristic_actions_device_async(
        env_p,
        profile_ids_device_ptr,
        out_actions_device_ptr,
        0
    );
    if (result != 0) {
        return result;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error("blackout_vec_heuristic_actions_device execution", err);
        return 1;
    }
    return 0;
}

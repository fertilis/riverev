#include <handlang/handlang.h>
#include <omp/omp.h>
#include "gpu_util.h"
#include "gpu_calc.h"
#include "util.h"
#include "riverev.h"

#define ALIGNED(x,y) (((x)%(y)) ? (x)+(y)-((x)%(y)) : x)

namespace riverev {
    
static float* HAND_COMPATIBILITY = nullptr; // 1326x1326

static void
calc_hand_compatibility_once();
    
static void
calc_ranking(const Board& board, float* out);

static BoardCompatibilty
calc_board_compatibility(const Board& board);

static Weights 
get_average_range(const std::vector<Weights>& ranges_);


Calculator::Calculator(IO* io, const Params& params)    
    : io(io)
    , params(params) 
{
    calc_hand_compatibility_once();
    board_compatibility = calc_board_compatibility(io->board);
    ranking = (float*)std::malloc(1326*1326*sizeof(float));
    calc_ranking(io->board, ranking);
}


Calculator::~Calculator() {
    if (ranking) {
        std::free(ranking);
        ranking = nullptr;
    }
    if (gpu_buffer) {
        util::gpu::cu_cudaFree(gpu_buffer);
        gpu_buffer = nullptr;
    }
}

void
Calculator::setup_gpu() 
{
    size_t offset = 0;
    
    hand_compatibility_offset = offset;
    size_t hand_compatibility_size = 1326 * 1326 * sizeof(float);
    offset += hand_compatibility_size;
    
    offset = ALIGNED(offset, 128);
    board_compatibility_offset = offset;
    size_t board_compatibility_size = 1326 * sizeof(float);
    offset += board_compatibility_size;
    
    offset = ALIGNED(offset, 128);
    ranking_offset = offset;
    size_t ranking_size = 1326 * 1326 * sizeof(float);
    offset += ranking_size;
    
    offset = ALIGNED(offset, 128);
    weights_offset = offset;
    size_t weights_size = 1326 * sizeof(float);
    offset += weights_size;
    
    offset = ALIGNED(offset, 128);
    values_offset = offset;
    size_t values_size = 1326 * sizeof(float);
    offset += values_size;
    
    offset = ALIGNED(offset, 128);
    gainsums_offset = offset;
    size_t gainsums_size = 1326 * sizeof(float);
    offset += gainsums_size;
    
    offset = ALIGNED(offset, 128);
    weightsums_offset = offset;
    size_t weightsums_size = 1326 * sizeof(float);
    offset += weightsums_size;
    
    offset = ALIGNED(offset, 128);
    size_t buffer_size = offset;
    
    util::gpu::cu_cublasCreate_once(params.gpu_device);
    util::gpu::cu_cudaMalloc((void**)&gpu_buffer, buffer_size);
    util::gpu::cu_cudaMemcpy_HD(
            gpu_buffer + hand_compatibility_offset, 
            HAND_COMPATIBILITY, 
            hand_compatibility_size);
    util::gpu::cu_cudaMemcpy_HD(
            gpu_buffer + board_compatibility_offset, 
            &board_compatibility[0], 
            board_compatibility_size);
    util::gpu::cu_cudaMemcpy_HD(
            gpu_buffer + ranking_offset, 
            ranking, 
            ranking_size);
    
    gpu_is_set_up = true;
}


void
Calculator::teardown_gpu()
{
    if (gpu_buffer) {
        util::gpu::cu_cudaFree(gpu_buffer);
        gpu_buffer = nullptr;
    }
}


void
Calculator::calc_showdown_values() 
{
    if (params.use_gpu && !params.always_montecarlo) {
        assert(gpu_is_set_up);
    }
    for (Node& node : io->nodes) {
        calc_node_values(node);
    }
}

void
Calculator::calc_node_values(Node& node)
{
    int n_players = node.active_players.size();
    assert(n_players >= 2);
    const int exp = n_players-1;
    for (int i = 0; i < n_players; i++) {
        const Player& hero = node.active_players[i];
        std::vector<Player> opponents = node.active_players;
        opponents.erase(opponents.begin()+i);
        std::vector<Weights> opponent_ranges;
        opponent_ranges.reserve(opponents.size());
        for (Player& opponent : opponents) {
            opponent_ranges.push_back(node.weightset[opponent.position]);
        }
        Values& values = node.valueset[hero.position];
        if ((n_players > 2 && !params.multiway_exponentiate) || params.always_montecarlo) {
            const Weights& hero_range = node.weightset[hero.position];
            values = calc_node_values_montecarlo(hero_range, opponent_ranges, hero.pot, hero.cost);
        } else {
            Weights opponent_weights = get_average_range(opponent_ranges);
            if (params.use_gpu) {
                values = calc_node_values_gpu(opponent_weights, hero.pot, hero.cost, exp);
            } else {
                values = calc_node_values_cpu(opponent_weights, hero.pot, hero.cost, exp);
            }
        }
    }
}


Values
Calculator::calc_node_values_cpu(const Weights& opponent_weights, float pot, float cost, int exp) 
{
    const int m = 1; 
    const int n = 1326; 
    const int k = 1326;
    const float* W = &opponent_weights[0];
    const float* R = ranking;
    const float* C = HAND_COMPATIBILITY;
    float A[1326];
    float B[1326];
    Values values;
    // Row-major matrix multiplications
    // W R = A (sum of weighted gains -- by opponent weights)
    // 1x1326 * 1136x1326  = 1x1326
    util::matmul_rowmajor<m, n, k>(W, R, A);
    // W C = B (sum of opponent weights)
    // 1x1326 * 1326x1326 = 1x1326
    util::matmul_rowmajor<m, n, k>(W, C, B);
    // V = (A had (1/B) * p - c
    for (int i = 0; i < 1326; i++) {
        float weightsum = B[i];
        float equity;
        if (weightsum) {
            equity = A[i]/weightsum;
            values[i] = std::pow(equity, exp) * pot - cost;
        } else {
            if (board_compatibility[i]) {
                values[i] = 0.0f;
            } else {
                values[i] = NaN;
            }
        }
    }
    return values;
}


Values
Calculator::calc_node_values_gpu(const Weights& opponent_weights, float pot, float cost, int exp) 
{
    util::gpu::cu_cudaMemcpy_HD(
            gpu_buffer + weights_offset, 
            opponent_weights.data(), 
            1326 * sizeof(float));
    util::gpu::cu_cudaDeviceSynchronize();
    
    gpucalc::calc_node_values(
            (const float*)(gpu_buffer + hand_compatibility_offset),
            (const float*)(gpu_buffer + board_compatibility_offset),
            (const float*)(gpu_buffer + ranking_offset),
            (const float*)(gpu_buffer + weights_offset),
            pot,
            cost,
            exp,
            (float*)(gpu_buffer + values_offset),
            (float*)(gpu_buffer + gainsums_offset),
            (float*)(gpu_buffer + weightsums_offset));
    util::gpu::cu_cudaDeviceSynchronize();
    
    Values values;
    util::gpu::cu_cudaMemcpy_DH(
            values.data(),
            gpu_buffer + values_offset, 
            1326 * sizeof(float));
    
    util::gpu::cu_cudaDeviceSynchronize();
    return values;
}


Values
Calculator::calc_node_values_montecarlo(const Weights& hero_range, const std::vector<Weights>& opponent_ranges, float pot, float cost)
{
    std::cout << "in mc" << std::endl;
    std::array<double,1326> equities = omp::calc_equity_distr(
            io->board.data(),
            hero_range, 
            opponent_ranges, 
            params.montecarlo_n_trials, 
            params.montecarlo_n_threads);
    Values values;
    for (int i = 0; i < 1326; i++) {
        if (board_compatibility[i] == 0.0f) {
            values[i] = NaN;
        } else {
            float equity = equities[i];
            if (equity < 0.0f) {
                values[i] = 0.0f;
            } else {
                values[i] = equity * pot - cost;
            }
        }
    }
    return values;
}


static void
calc_hand_compatibility_once()
{
    if (HAND_COMPATIBILITY) {
        return;
    }
    HAND_COMPATIBILITY = (float*)std::malloc(1326*1326*sizeof(float));
    std::array<uint64_t, 1326> handmasks;
    for (int i = 0; i < 1326; i++) {
        handmasks[i] = hl::cardmask(hl::hands[i]);
    }
    for (int i = 0; i < 1326; i++) {
        for (int j = 0; j < 1326; j++) {
            if (handmasks[i] & handmasks[j]) {
                *(HAND_COMPATIBILITY + i*1326 + j) = 0.0f;
            } else {
                *(HAND_COMPATIBILITY + i*1326 + j) = 1.0f;
            }
        }
    }
}


static void
calc_ranking(const Board& board, float* ranking) 
{
    // Calc ranks
    std::array<uint16_t,1326> ranks;
    omp::HandEvaluator eval;
    omp::Hand hand5 = omp::Hand::empty();
    for (const uint8_t& card : board) {
        assert(card < 52);
        hand5 += omp::Hand((unsigned)card);
    }
    uint64_t boardmask = hl::cardmask(board);
    for (int hi = 0; hi < 1326; hi++) {
        hl::Hand hand = hl::hands[hi];
        uint64_t handmask = hl::cardmask(hand);
        if (boardmask & handmask) {
            ranks[hi] = 0;
            continue;
        }
        omp::Hand hand7(hand5);
        hand7 += omp::Hand((unsigned)hand[0]);
        hand7 += omp::Hand((unsigned)hand[1]);
        ranks[hi] = eval.evaluate(hand7);
    }
    // Rows will become columns in matmul
    // Operation will be Ranking x Weights in column-major format
    // with Weights of 1326x1 dimension.
    // So, in row-major format, a row of ranking represents hero gains
    uint16_t hero_rank, opp_rank;
    for (int hero_hi = 0; hero_hi < 1326; hero_hi++) {
        hero_rank = ranks[hero_hi];
        for (int opp_hi = 0; opp_hi < 1326; opp_hi++) {
            opp_rank = ranks[opp_hi];
            if (hero_rank > opp_rank) {
                *(ranking + hero_hi*1326 + opp_hi) = 1.0f;
            } else if (hero_rank == opp_rank) {
                if (hero_rank == 0) { // incompatible hands don't add up to ev
                    *(ranking + hero_hi*1326 + opp_hi) = 0.0f; 
                } else {
                    *(ranking + hero_hi*1326 + opp_hi) = 0.5f;
                }
            } else {
                *(ranking + hero_hi*1326 + opp_hi) = 0.0f;
            }
        }
    }
}


static BoardCompatibilty
calc_board_compatibility(const Board& board) 
{
    BoardCompatibilty compatibility;
    uint64_t boardmask = hl::cardmask(board);
    for (int i = 0; i < 1326; i++) {
        if (boardmask & hl::cardmask(hl::hands[i])) {
            compatibility[i] = 0.0f;
        } else {
            compatibility[i] = 1.0f;
        }
    }
    return compatibility;
}


static Weights 
get_average_range(const std::vector<Weights>& ranges_) 
{
    Weights out;
    int n = ranges_.size();
    for (int hi = 0; hi < 1326; hi++) {
        float weightsum = 0.0f;
        for (const Weights& range_ : ranges_) {
            weightsum += range_[hi];
        }
        out[hi] = weightsum/n;
    }
    return out;
}

} // riverev

#pragma once
#include <cstdint>
#include <limits>
#include <vector>
#include <array>
#include "evio.h"


namespace riverev {
    
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

using Board = evio::Board;
using Weights = evio::Weights;
using Weightset = evio::Weightset;
using Values = evio::Values;
using Valueset = evio::Valueset;
using Player = evio::Player;
using Node = evio::Node;
using IO = evio::IO;

using Ranking = std::array<std::array<float,1326>, 1326>; // 1326x1326
using BoardCompatibilty = std::array<float,1326>; 


struct Params {
    bool use_gpu = true;
    bool multiway_exponentiate = true; // false: monte-carlo
    bool always_montecarlo = false;
    unsigned montecarlo_n_trials = 10000;
    unsigned montecarlo_n_threads = 6;
    int gpu_device = 0;
};


class Calculator {
public:
    Calculator(IO* io, const Params& params);
    ~Calculator();
    void setup_gpu();
    void teardown_gpu();
    void calc_showdown_values();
    
private:
    IO* io;
    Params params;
    
    float* total_compatibility = nullptr; // 1326x1326
    float* ranking = nullptr; // 1326x1326
    BoardCompatibilty board_compatibility;
    
    bool gpu_is_set_up = false;
    char* gpu_buffer = nullptr;
    // offsets in gpu_buffer
    size_t hand_compatibility_offset;
    size_t board_compatibility_offset;
    size_t ranking_offset;
    size_t weights_offset;
    size_t values_offset;
    size_t gainsums_offset;
    size_t weightsums_offset;
    
    void calc_node_values(Node& node);
    Values calc_node_values_cpu(const Weights& opponent_weights, float pot, float cost, int exp); 
    Values calc_node_values_gpu(const Weights& opponent_weights, float pot, float cost, int exp);
    Values calc_node_values_montecarlo(const Weights& hero_range, const std::vector<Weights>& opponent_ranges, float pot, float cost);
};

} // riverev

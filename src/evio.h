/**
 * Common input interface for value calculations (inferences)
 */
#pragma once
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>

namespace evio {
    
constexpr int MAX_PLAYERS = 10;
using Board = std::array<uint8_t,5>;
using Weights = std::array<float,1326>;
using Weightset = std::array<Weights,MAX_PLAYERS>;
using Values = std::array<float,1326>;
using Valueset = std::array<Values,MAX_PLAYERS>;

struct Player {
    int position;
    float pot;
    float cost;
};

struct Node {
    Weightset weightset; // input 
    Valueset valueset;  // output
    std::vector<Player> active_players;
};

struct IO {
    Board board;
    std::vector<Node> nodes;
};

} // evio

#include <iostream>
#include <handlang/handlang.h>
#include <riverev/riverev.h>
#include "../src/util.h"

using namespace riverev;

void try_mc() {
   IO io;
    io.board = hl::to_board("AsTc8d6h3s");
    Node& node = io.nodes.emplace_back();
    node.active_players.emplace_back(0, 1.0, 0.0);
    node.active_players.emplace_back(2, 1.0, 0.0);
    node.weightset[0] = hl::itof_range(hl::to_range("x22"));
    node.weightset[2] = hl::itof_range(hl::to_range("x22"));

    Params params;
    params.use_gpu = false;
    params.always_montecarlo = false;

    Calculator calc(&io, params);

    calc.calc_showdown_values();
    
    util::print_array(node.valueset[0], 1326);
}


int main() {
    try_mc();
}

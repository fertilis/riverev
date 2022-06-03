#include <json.h> // nlohmann::json
#include "riverev.h"
#include "evio.h"


using json = nlohmann::json;
using namespace evio;
using namespace riverev;

extern "C" {
    
Params* new_params() {
    return new Params;
}

void delete_params(Params* params)  {
    delete params;
}
    
IO* new_io() {
    return new IO;
}

void delete_io(IO* io)  {
    delete io;
}

void add_node(IO* io) {
    io->nodes.emplace_back();
}

Node* get_node(IO* io, int index) {
    return &(io->nodes[index]);
}

void add_player(Node* node) {
    node->active_players.emplace_back();
}

Player* get_player(Node* node, int index) {
    return &(node->active_players[index]);
}

Calculator* new_calculator(IO* io, const Params* params) {
    return new Calculator(io, *params);
}

void delete_calculator(Calculator* calc)  {
    delete calc;
}

void setup_gpu(Calculator* calc) {
    calc->setup_gpu();
}

void calc_showdown_values(Calculator* calc) {
    calc->calc_showdown_values();
}
    
} // C

#include <iostream>
#include <handlang/handlang.h>
#include <riverev/riverev.h>
#include "../src/util.h"
#include "../src/timeit.h"

using namespace riverev;
using namespace util::timeit;

void try_calc() {
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
    std::cout << std::endl;
    
    params.always_montecarlo = true;
    Calculator calc2(&io, params);
    calc2.calc_showdown_values();
    util::print_array(node.valueset[0], 1326);
    std::cout << std::endl;
    
    params.use_gpu = true;
    params.always_montecarlo = false;
    Calculator calc3(&io, params);
    calc3.setup_gpu();
    calc3.calc_showdown_values();
    util::print_array(node.valueset[0], 1326);
    std::cout << std::endl;
}

void time_calc() {
    IO io;
    io.board = hl::to_board("AsTc8d6h3s");
    for (int i = 0; i < 1000; i++) {
        Node& node = io.nodes.emplace_back();
        node.active_players.emplace_back(0, 1.0, 0.0);
        node.active_players.emplace_back(2, 1.0, 0.0);
        node.weightset[0] = hl::itof_range(hl::to_range("x22"));
        node.weightset[2] = hl::itof_range(hl::to_range("x22"));
    }

    Params params;
    params.use_gpu = false;
    
    Calculator calc(&io, params);
    timeit(0);
    calc.calc_showdown_values();
    timeit(1, 'm', "cpu");
    
    params.use_gpu = true;
    Calculator calc3(&io, params);
    calc3.setup_gpu();
    timeit(0);
    calc3.calc_showdown_values();
    timeit(1, 'm', "gpu");
}


struct TestCase {
    std::string board;
    std::string opp_range;
    std::string hand;
    float pot;
    float cost;
    float value;
};

std::vector<TestCase> TEST_CASES = {
    {"AsTc8d6h3s", "x22", "2s2c", 1.0, 0.0, 0.3804},
    {"QsTcJd6h3s", "x22", "2s2c", 1.0, 0.0, 0.3479},
    {"AsTc8d6h3s", "x22", "KsJc", 1.0, 0.0, 0.3641},
    {"AsTc8d6h3s", "x22", "8s5c", 1.0, 0.0, 0.6282},
    {"AsTc8d6h3s", "55+ AJT xQT", "8s5c", 1.0, 0.0, 0.2897},
    {"AsTc8d6h3s", "55+ AJT xQT", "8s5c", 10.5, 12.0, 0.2897*10.5-12.0},
    {"AsTc8d6h3s", "x22", "8d5c", 1.0, 0.0, riverev::NaN},
    {"AsTc8d8h3s", "88", "8s6c", 1.0, 0.0, 0.0},
    {"AsTc8d6h3s", "x22", "2s2c", 1.0, 0.0, 0.3804},
    {"AsTc8d8h3s", "88", "8s6c", 1.0, 0.0, 0.0},
};

std::vector<TestCase> TEST_CASES_SAME_BOARD = {
    {"AsTc8d6h3s", "x22", "2s2c", 1.0, 0.0, 0.3804},
    {"AsTc8d6h3s", "x22", "KsJc", 1.0, 0.0, 0.3641},
    {"AsTc8d6h3s", "x22", "8s5c", 1.0, 0.0, 0.6282},
    {"AsTc8d6h3s", "55+ AJT xQT", "8s5c", 1.0, 0.0, 0.2897},
    {"AsTc8d6h3s", "55+ AJT xQT", "8s5c", 10.5, 12.0, 0.2897*10.5-12.0},
    {"AsTc8d6h3s", "x22", "8d5c", 1.0, 0.0, riverev::NaN},
};


void test_calculations() {
    IO io;
    Node& node = io.nodes.emplace_back();
    node.active_players.emplace_back(0, 1.0, 0.0);
    node.active_players.emplace_back(2, 1.0, 0.0);
    node.weightset[0] = hl::itof_range(hl::to_range("x22"));

    Params params;
    float tol = 0.01;
    for (bool use_gpu : std::vector<bool>{false, true}) {
        std::cout << "GPU: " << use_gpu << std::endl;
        
        params.use_gpu = use_gpu;
        for (auto& test_case : TEST_CASES) {
            io.board = hl::to_board(test_case.board);
            node.active_players[0].pot = test_case.pot;
            node.active_players[0].cost = test_case.cost;
            node.weightset[2] = hl::itof_range(hl::to_range(test_case.opp_range));
            Calculator calc(&io, params);
            if (use_gpu) {
                calc.setup_gpu();
            }
            calc.calc_showdown_values();
            int hand_index = hl::hand_index(hl::to_hand(test_case.hand));
            float value = node.valueset[0][hand_index];
            if (util::almost_equal(value/test_case.pot, test_case.value/test_case.pot, tol)) {
                std::cout << "ok: " << test_case.board << " : " << test_case.opp_range << " : " << test_case.hand << 
                    " : " << util::round_to(test_case.value, 4) << " : " << util::round_to(value, 4) << std::endl;
            } else {
                std::cout << "fail: " << test_case.board << " : " << test_case.opp_range << " : " << test_case.hand << 
                    " : " << util::round_to(test_case.value, 4) << " : " << util::round_to(value, 4) << std::endl;
            }
        }
        std::cout << std::endl;
    }
    for (bool use_gpu : std::vector<bool>{false, true}) {
        std::cout << "GPU: " << use_gpu << std::endl;
        
        io.board = hl::to_board(TEST_CASES_SAME_BOARD[0].board);
        params.use_gpu = use_gpu;
        Calculator calc(&io, params);
        if (use_gpu) {
            calc.setup_gpu();
        }
        for (auto& test_case : TEST_CASES_SAME_BOARD) {
            node.active_players[0].pot = test_case.pot;
            node.active_players[0].cost = test_case.cost;
            node.weightset[2] = hl::itof_range(hl::to_range(test_case.opp_range));
            calc.calc_showdown_values();
            int hand_index = hl::hand_index(hl::to_hand(test_case.hand));
            float value = node.valueset[0][hand_index];
            if (util::almost_equal(value/test_case.pot, test_case.value/test_case.pot, tol)) {
                std::cout << "ok: " << test_case.board << " : " << test_case.opp_range << " : " << test_case.hand << 
                    " : " << util::round_to(test_case.value, 4) << " : " << util::round_to(value, 4) << std::endl;
            } else {
                std::cout << "fail: " << test_case.board << " : " << test_case.opp_range << " : " << test_case.hand << 
                    " : " << util::round_to(test_case.value, 4) << " : " << util::round_to(value, 4) << std::endl;
            }
        }
        std::cout << std::endl;
    }
    
    
}

int main() {
    test_calculations();
}

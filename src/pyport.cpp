#include <evio/evio.h> // evio
#include "riverev.h"


using namespace riverev;

extern "C" {
    
Params* new_params() {
    return new Params;
}

void delete_params(Params* params)  {
    delete params;
}
    
Calculator* new_calculator(evio::IO* io, const Params* params) {
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

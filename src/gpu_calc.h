#pragma once


namespace riverev::gpucalc {
    
void
calc_node_values(
        const float* hand_compatibility,
        const float* board_compatibility,
        const float* ranking,
        const float* opponent_weights,
        const float pot,
        const float cost,
        const float exp,
        float* values,
        float* gainsums,
        float* weightsums);

} // riverev::gpucalc

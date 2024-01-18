# River EV Calculator

Calculates river values exactly (cpu and gpu calculations) or approximately (monte-carlo simulation by omp module).
If a node has more than two active players, exact calculations are done versus average range by equity exponentiation.
If there are more than one opponent and their ranges are quite narrow, monte-carlo method will be more accurate.

GPU calculations are about 20 times faster than on CPU.

## Dependencies

Build: `libevio.a`, `libhandlang.a`, `libomp.a`, cuda sdk (`nvcc`)

To install cuda sdk on ubuntu 22.04 (3.5 Gb download):

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -P /tmp  || exit 1
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb || exit 1
sudo apt update || exit 1
sudo apt install -y cuda || exit 1
add_to_bashrc "add_to_path /usr/local/cuda/bin"
add_to_bashrc "add_to_ld_library_path /usr/local/cuda-12.2/lib64"
```

Runtime: `libcublas`, `libcudart`

## Build

```bash
make clean
make
make install
make tests
_build/test
```

## C++ API

```C++
#include <iostream>
#include <handlang/handlang.h>
#include <riverev/riverev.h>

int main() {
    Params params;
    params.use_gpu = true;

    IO io;
    io.board = hl::to_board("AsTc8d6h3s");
    Node& node = io.nodes.emplace_back();
    node.active_players.emplace_back(0, 1.0, 0.0);
    node.active_players.emplace_back(2, 1.0, 0.0);
    node.weightset[0] = hl::itof_range(hl::to_range("x22"));
    node.weightset[2] = hl::itof_range(hl::to_range("x22"));

    Calculator calc(&io, params);
    calc.setup_gpu();
    calc.calc_showdown_values();
    std::cout << node.valueset[hl::hand_index(hl::to_hand("8s5c"))] << std::endl; // 0.6282
    node.weightset[2] = hl::itof_range(hl::to_range("55+ AJT xQT"));
    calc.calc_showdown_values();
    std::cout << node.valueset[hl::hand_index(hl::to_hand("8s5c"))] << std::endl; // 0.2897
}
```

## Python API

```python
import handlang as hl
import numpy as np
import riverev as rv


params = rv.Params()
params.use_gpu = True

io = rv.IO('AsTc8d6h3s')

node = io.add_node()
player = node.add_player()
player.position = 0
player.pot = 10.5
player.cost = 12.0
node.weightset[0] = hl.to_range("x22").astype(np.float32)/100

player = node.add_player()
player.position = 2
player.pot = 10.5
player.cost = 12.0
node.weightset[2] = hl.to_range("55+ AJT xQT").astype(np.float32)/100

calc = rv.Calculator(io, params)
calc.setup_gpu()
calc.calc_showdown_values()

print(node.valueset[0][hl.hand_index(hl.to_hand('8s5c'))])
print(0.2897*10.5-12.0)
```

import handlang as hl
import numpy as np
import evio as ev
import riverev as rv


params = rv.Params()
params.use_gpu = True

io = ev.IO.new()
io.board = hl.to_board('AsTc8d6h3s')

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


io = ev.IO.new()
io.board = hl.to_board('AsTc8d6h3s')

node = io.add_node()
player = node.add_player()
player.position = 0
player.pot = 1.0
player.cost = 0.0
node.weightset[0] = hl.to_range("x22").astype(np.float32)/100

player = node.add_player()
player.position = 1
player.pot = 1.0
player.cost = 0.0
node.weightset[1] = hl.to_range("x22").astype(np.float32)/100

player = node.add_player()
player.position = 2
player.pot = 1.0
player.cost = 0.0
node.weightset[2] = hl.to_range("x22").astype(np.float32)/100

calc = rv.Calculator(io, params)
calc.setup_gpu()
calc.calc_showdown_values()

print(node.valueset[0][hl.hand_index(hl.to_hand('2s2c'))])
print(0.3804**2)

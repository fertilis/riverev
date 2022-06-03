from ctypes import (
    Structure,
    c_void_p,
    c_ubyte,
    c_int,
    c_float,
    c_bool,
    c_uint,
    POINTER,
    cast,
)
import numpy as np
import ctypes
import os

import sys
#np.set_printoptions(threshold=sys.maxsize)

import handlang as hl

    
class _StructureWrap:
    def __init__(self, getptr):
        self._getptr = getptr
        ptr = self._getptr()
        self._fields = [x[0] for x in ptr.contents._fields_]
        
    def __getattr__(self, name):
        if name in self._fields:
            return getattr(self._getptr().contents, name)
        else:
            return object.__getattribute__(self, name)
    
    def __setattr__(self, name, value):
        if name in ['_getptr', '_fields', '_pointer']:
            object.__setattr__(self, name, value)
        elif name in self._fields:
            setattr(self._getptr().contents, name, value)
        else:
            object.__setattr__(self, name, value)
        
        

class _Player(Structure):
    _fields_ = [
        ('position', c_int),
        ('pot', c_float),
        ('cost', c_float),
    ]
    
    
class _Node(Structure):
    _fields_ = [
        ('_weightset', c_float*10*1326),
        ('_valueset', c_float*10*1326),
    ]
        
class _IO(Structure):
    _fields_ = [
        ('board', c_ubyte*5),
    ]
    
    
class _Params(Structure):
    _fields_ = [
        ('use_gpu', c_bool),
        ('multiway_exponentiate', c_bool),
        ('always_montecarlo', c_bool),
        ('montecarlo_n_trials', c_uint),
        ('montecarlo_n_threads', c_uint),
        ('gpu_device', c_int),
    ]
    
    
class Player(_StructureWrap): pass
    

class Node(_StructureWrap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_players = []
        
    def add_player(self)->Player:
        _lib.add_player(self._getptr())
        index = len(self.active_players)
        ptr_getter = lambda index=index: (
            cast(_lib.get_player(self._getptr(), index), POINTER(_Player))
        )
        player = Player(ptr_getter)
        self.active_players.append(player)
        return player
    
    @property
    def weightset(self):
        return np.frombuffer(
            self._getptr().contents._weightset, np.float32).reshape((10, 1326))
    
    @property
    def valueset(self):
        return np.frombuffer(
            self._getptr().contents._valueset, np.float32).reshape((10, 1326))
        

class IO:
    def __init__(self, sboard:str):
        vp = _lib.new_io()
        self._pointer = cast(vp, POINTER(_IO))
        for i, card in enumerate(hl.to_board(sboard)):
            self._pointer.contents.board[i] = card
        self.nodes = []
        
    def add_node(self)->Node:
        _lib.add_node(self._pointer)
        index = len(self.nodes)
        ptr_getter = lambda index=index: (
            cast(_lib.get_node(self._pointer, index), POINTER(_Node))
        )
        node = Node(ptr_getter)
        self.nodes.append(node)
        return node
    
    def __del__(self):
        _lib.delete_io(self._pointer)
    

class Params(_StructureWrap):
    def __init__(self):
        vp = _lib.new_params()
        self._pointer = cast(vp, POINTER(_Params))
        self._getptr = lambda: self._pointer
        self._fields = [x[0] for x in self._pointer.contents._fields_]
    
    def __del__(self):
        _lib.delete_params(self._pointer)
        
    
class Calculator:
    def __init__(self, io: IO, params: Params):
        self._pointer = _lib.new_calculator(io._pointer, params._pointer)
    
    def __del__(self):
        _lib.delete_calculator(self._pointer)
        
    def setup_gpu(self):
        _lib.setup_gpu(self._pointer)
        
    def calc_showdown_values(self):
        _lib.calc_showdown_values(self._pointer)



_lib = None
def _init_lib():
    global _lib
    if _lib is not None:
        return
    path = os.path.abspath(__file__+'/../_riverev.so')
    _lib = ctypes.cdll.LoadLibrary(path)
    
    _lib.new_params.argtypes = []
    _lib.new_params.restype = c_void_p
    _lib.delete_params.argtypes = [c_void_p]
    
    _lib.new_io.argtypes = []
    _lib.new_io.restype = c_void_p
    _lib.delete_io.argtypes = [c_void_p]
    
    _lib.new_calculator.argtypes = []
    _lib.new_calculator.restype = c_void_p
    _lib.delete_calculator.argtypes = [c_void_p]
    
    _lib.add_node.argtypes = [c_void_p]
    _lib.get_node.argtypes = [c_void_p, c_int]
    _lib.get_node.restype = c_void_p
    
    _lib.add_player.argtypes = [c_void_p]
    _lib.get_player.argtypes = [c_void_p, c_int]
    _lib.get_player.restype = c_void_p
    
    _lib.setup_gpu.argtypes = [c_void_p]
    _lib.calc_showdown_values.argtypes = [c_void_p]
    
_init_lib()

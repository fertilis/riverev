from ctypes import (
    cdll,
    Structure,
    c_void_p,
    c_int,
    c_bool,
    c_uint,
    POINTER,
    cast,
)
import numpy as np
import os

    
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
        

class _Params(Structure):
    _fields_ = [
        ('use_gpu', c_bool),
        ('multiway_exponentiate', c_bool),
        ('always_montecarlo', c_bool),
        ('montecarlo_n_trials', c_uint),
        ('montecarlo_n_threads', c_uint),
        ('gpu_device', c_int),
    ]
    
    
class Params(_StructureWrap):
    def __init__(self):
        vp = _lib.new_params()
        self._pointer = cast(vp, POINTER(_Params))
        self._getptr = lambda: self._pointer
        self._fields = [x[0] for x in self._pointer.contents._fields_]
    
    def __del__(self):
        _lib.delete_params(self._pointer)
        
    
class Calculator:
    def __init__(self, io: 'evio.IO', params: Params):
        self._pointer = _lib.new_calculator(io.pointer, params._pointer)
    
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
    _lib = cdll.LoadLibrary(path)
    
    _lib.new_params.argtypes = []
    _lib.new_params.restype = c_void_p
    _lib.delete_params.argtypes = [c_void_p]
    
    _lib.new_calculator.argtypes = []
    _lib.new_calculator.restype = c_void_p
    _lib.delete_calculator.argtypes = [c_void_p]
    
    _lib.setup_gpu.argtypes = [c_void_p]
    _lib.calc_showdown_values.argtypes = [c_void_p]
    
_init_lib()

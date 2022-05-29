PROJECT = riverev
VERSION = 1.0.0

SSE4=1
CXX=g++-11
CXXFLAGS += -O3 -std=gnu++2b -Wall -Wpedantic -fPIC -Ideps -g
LINK = deps/omp/omp.a deps/handlang/handlang.a -L/usr/local/cuda/lib64 -lcublas -lcudart

ifdef SYSTEMROOT
    CXXFLAGS += -lpthread
else
    CXXFLAGS += -pthread
endif

ifeq ($(SSE4),1)
    CXXFLAGS += -msse4.2
endif

SOURCES := $(wildcard src/*.cpp)
BASE_NAMES := $(basename $(notdir $(SOURCES)))

OBJECTS := $(addprefix _build/,$(addsuffix .o,${BASE_NAMES}))
PY_OBJECTS := $(OBJECTS)
CPP_OBJECTS := $(filter-out _build/pyport.o, $(OBJECTS))
CUDA_OBJECTS := _build/gpu_util.o _build/gpu_calc.o


all: dirs $(CUDA_OBJECTS) $(OBJECTS) $(PROJECT).a #py$(PROJECT).a py$(PROJECT).so

.PHONY: all tests

dirs:
	mkdir -p _build _dist/$(PROJECT)

$(foreach s,$(BASE_NAMES),$(info New rule: _build/$s.o: src/$s.cpp)$(eval _build/$s.o: src/$s.cpp))

$(OBJECTS): 
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PROJECT).a: $(CPP_OBJECTS) $(CUDA_OBJECTS)
	ar rcs _dist/$(PROJECT)/$(PROJECT).a $^
	cp src/$(PROJECT).h _dist/$(PROJECT)/
	cp src/evio.h _dist/$(PROJECT)/
	echo $(VERSION) > _dist/$(PROJECT)/_version
	echo $(VERSION) > py/_version

py$(PROJECT).a:
	ar rcs _build/py$(PROJECT).a $(PY_OBJECTS) $(CUDA_OBJECTS)

py$(PROJECT).so: 
	gcc -Wl,--whole-archive _build/py$(PROJECT).a $(LINK) -Wl,--no-whole-archive -o py/_$(PROJECT).so -shared -fPIC -I/usr/include/python3.10  -lstdc++

clean:
	rm -rf _build _dist py/_$(PROJECT).so

install:
	rm -rf ~/dist/python/$(PROJECT)
	cp -rfa py ~/dist/python/$(PROJECT)

tests:
	$(CXX) $(CXXFLAGS) -I_dist tests/test.cpp _dist/$(PROJECT)/$(PROJECT).a $(LINK) -o _build/test

try:
	$(CXX) $(CXXFLAGS) -I_dist tests/try.cpp _dist/$(PROJECT)/$(PROJECT).a $(LINK) -o _build/try
	
	
_build/gpu_util.o:	
	nvcc -c ./src/gpu_util.cu --compiler-options '-fPIC' -o ./_build/gpu_util.o 
	
_build/gpu_calc.o:	
	nvcc -c ./src/gpu_calc.cu --compiler-options '-fPIC' -o ./_build/gpu_calc.o 

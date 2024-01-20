PROJECT := riverev
PYTHON_VERSION ?= 3.12
GPP_VERSION ?= 11
INCLUDE_DIR ?= $(shell pwd)/../../dist/include
LINK_DIR ?= $(shell pwd)/../../dist/lib
STATIC_LIB := _build/lib$(PROJECT).a
DYNAMIC_LIB := py/$(PROJECT)/_$(PROJECT).so
CXX := g++-$(GPP_VERSION)
CXXFLAGS := -O3 -std=gnu++2b -Wall -Wpedantic -fPIC -I$(INCLUDE_DIR)
DYNAMIC_LINKS := -L/usr/local/cuda/lib64 -lcudart -lcublas
TEST_LINKS := $(STATIC_LIB) -L$(LINK_DIR) -lhandlang -lomp -levio $(DYNAMIC_LINKS)
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:src/%.cpp=_build/%.o)
OBJECTS := $(filter-out _build/pyport.o, $(OBJECTS))
PORT_OBJECT := _build/pyport.o
INCLUDES := _build/include/$(PROJECT)/riverev.h
CUDA_OBJECTS := _build/gpu_util.o _build/gpu_calc.o

all: check_dependencies dirs $(STATIC_LIB) $(DYNAMIC_LIB) $(INCLUDES)
	
.PHONY: all clean tests try
	
check_dependencies:
	@if [ ! -f $(LINK_DIR)/libhandlang.a ]; then exit 1; fi
	@if [ ! -f $(LINK_DIR)/libomp.a ]; then exit 1; fi
	@if [ ! -f $(INCLUDE_DIR)/evio/evio.h ]; then exit 1; fi
	@if [ ! -f /usr/local/cuda/lib64/libcudart.so ]; then exit 1; fi
	@if [ ! -f /usr/local/cuda/lib64/libcublas.so ]; then exit 1; fi
	
	
dirs:
	@mkdir -p _build/include/$(PROJECT)
	
_build/gpu_util.o:	
	nvcc -c ./src/gpu_util.cu --compiler-options '-fPIC' -o ./_build/gpu_util.o 
	
_build/gpu_calc.o:	
	nvcc -c ./src/gpu_calc.cu --compiler-options '-fPIC' -o ./_build/gpu_calc.o 

_build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
$(STATIC_LIB): $(OBJECTS) $(CUDA_OBJECTS)
	ar rcs $@ $^
	
_build/include/$(PROJECT)/%.h: src/%.h
	cp $< $@
	
$(DYNAMIC_LIB): $(STATIC_LIB) $(PORT_OBJECT)
	gcc $(PORT_OBJECT) \
		-Wl,--whole-archive \
		$(STATIC_LIB) \
		$(LINK_DIR)/libhandlang.a \
		$(LINK_DIR)/libomp.a \
		-Wl,--no-whole-archive \
		$(DYNAMIC_LINKS) \
		-I$(INCLUDE_DIR) \
		-o $@ -shared -fPIC -I/usr/include/python$(PYTHON_VERSION)  -lstdc++

clean:
	rm -rf _build $(DYNAMIC_LIB)
	rm -rf py/*.egg-info
	rm -rf py/$(PROJECT)/__pycache__


tests:
	$(CXX) $(CXXFLAGS) -I_build/include tests/test.cpp $(TEST_LINKS) -o _build/test


#
#PROJECT = riverev
#VERSION = 1.0.0
#
#SSE4=1
#CXX=g++-11
#CXXFLAGS += -O3 -std=gnu++2b -Wall -Wpedantic -fPIC -Ideps
#LINK = deps/omp/omp.a deps/handlang/handlang.a -L/usr/local/cuda/lib64 -lcublas -lcudart
#
#ifdef SYSTEMROOT
#    CXXFLAGS += -lpthread
#else
#    CXXFLAGS += -pthread
#endif
#
#ifeq ($(SSE4),1)
#    CXXFLAGS += -msse4.2
#endif
#
#SOURCES := $(wildcard src/*.cpp)
#BASE_NAMES := $(basename $(notdir $(SOURCES)))
#
#OBJECTS := $(addprefix _build/,$(addsuffix .o,${BASE_NAMES}))
#PY_OBJECTS := $(OBJECTS)
#CPP_OBJECTS := $(filter-out _build/pyport.o, $(OBJECTS))
#CUDA_OBJECTS := _build/gpu_util.o _build/gpu_calc.o
#
#
#all: dirs $(CUDA_OBJECTS) $(OBJECTS) $(PROJECT).a py$(PROJECT).a py$(PROJECT).so
#
#.PHONY: all tests
#
#dirs:
#	mkdir -p _build _dist/$(PROJECT)
#
#$(foreach s,$(BASE_NAMES),$(info New rule: _build/$s.o: src/$s.cpp)$(eval _build/$s.o: src/$s.cpp))
#
#$(OBJECTS): 
#	$(CXX) $(CXXFLAGS) -c $< -o $@
#
#$(PROJECT).a: $(CPP_OBJECTS) $(CUDA_OBJECTS)
#	ar rcs _dist/$(PROJECT)/$(PROJECT).a $^
#	cp src/$(PROJECT).h _dist/$(PROJECT)/
#	echo $(VERSION) > _dist/$(PROJECT)/_version
#	echo $(VERSION) > py/_version
#
#py$(PROJECT).a:
#	ar rcs _build/py$(PROJECT).a $(PY_OBJECTS) $(CUDA_OBJECTS)
#
#py$(PROJECT).so: 
#	gcc -Wl,--whole-archive _build/py$(PROJECT).a $(LINK) -Wl,--no-whole-archive -o py/_$(PROJECT).so -shared -fPIC -I/usr/include/python3.10  -lstdc++
#
#clean:
#	rm -rf _build _dist py/_$(PROJECT).so
#
#install:
#	rm -rf ~/dist/python/$(PROJECT)
#	cp -rfa py ~/dist/python/$(PROJECT)
#
#tests:
#	$(CXX) $(CXXFLAGS) -I_dist tests/test.cpp _dist/$(PROJECT)/$(PROJECT).a $(LINK) -o _build/test
#
#try:
#	$(CXX) $(CXXFLAGS) -I_dist tests/try.cpp _dist/$(PROJECT)/$(PROJECT).a $(LINK) -o _build/try
#	
#	
#_build/gpu_util.o:	
#	nvcc -c ./src/gpu_util.cu --compiler-options '-fPIC' -o ./_build/gpu_util.o 
#	
#_build/gpu_calc.o:	
#	nvcc -c ./src/gpu_calc.cu --compiler-options '-fPIC' -o ./_build/gpu_calc.o 

include Makefile.inc

CXX        = g++
OPTFLAGS   = -std=c++17 -O3 -D$(LIB_SELECT) -DLIB_GINKGO -DLIB_PETSC
BIN        = build/spmv_driver  
COMMON_OBJ = src/utils.cpp src/driver_spmv.cpp

ifeq ($(LIB_SELECT), LIB_GINKGO)
  #openmpi need it? -lmpi_cxx
  LIB     = -L/lib/x86_64-linux-gnu/ -L$(GINKGO_DIR)/build/lib -lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -lginkgo_dpcpp -lginkgo_device -lmpi -lmpi_cxx
  INCLUDE = -I$(GINKGO_DIR)/include/ -I$(GINKGO_DIR)/build/include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/
  OBJ     = $(COMMON_OBJ)
else
  LIB     = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -L/lib/x86_64-linux-gnu/ -L$(GINKGO_DIR)/build/lib -lpetsc -lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -lginkgo_dpcpp -lginkgo_device -lmpi -lmpi_cxx
  INCLUDE = -I$(PETSC_DIR)/src/mat/tests/ -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(GINKGO_DIR)/include/ -I$(GINKGO_DIR)/build/include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/
  OBJ     = $(PETSC_DIR)/src/mat/tests/mmloader.c $(PETSC_DIR)/src/mat/tests/mmio.c $(COMMON_OBJ)
endif

#g++ -std=c++17 -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/home/hmartinez/software/ginkgo/include/ -I/home/hmartinez/software/ginkgo/build/include/ src/utils.cpp src/driver_spmv.cpp -o build/spmv_driver -L/home/hmartinez/software/ginkgo/build/lib -L/lib/x86_64-linux-gnu/libmpi/ -lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -lginkgo_dpcpp -lginkgo_device -lmpi -lmpi_cxx

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(OPTFLAGS) $(INCLUDE) $^ -o $@ $(LIB)
clean:
	rm build/*

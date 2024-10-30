
FC = h5pfc
SELF_ROOT ?= /usr/local/self
FEQ_PARSE_ROOT ?= /usr/local/self

LIBS = "${SELF_ROOT}/lib/libself-static.a -L${FEQ_PARSE_ROOT}/lib -lfeqparse"
INCLUDES = "-I${SELF_ROOT}/include"

# # If building with GPU accelerated libraries
LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lhiprtc -lhipblas 
INCLUDES += $(hipconfig -C) -I${MAGMA_ROOT}/include

divergence_benchmarks:
	hipcc -c -O3 kernels.cpp -o kernels.o
	${FC} -c -O3 -fopenmp -ffree-line-length-512 divergence_3d.f90 ${INCLUDES} ${LIBS} -o divergence_3d.o
	${FC} -fopenmp -ffree-line-length-512 kernels.o divergence_3d.o ${INCLUDES} ${LIBS} -o $@

.PHONY: clean

clean:
	rm -f divergence_benchmarks *.o *.mod
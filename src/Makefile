
FC = h5pfc
SELF_ROOT ?= /usr/local/self
FEQ_PARSE_ROOT ?= /usr/local/self
CXX = hipcc


LIBS = "${SELF_ROOT}/lib/libself-static.a -L${FEQ_PARSE_ROOT}/lib -lfeqparse"
INCLUDES = "-I${SELF_ROOT}/include"

# # If building with GPU accelerated libraries (AMD ROCm/HIP)
LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lhiprtc -lhipblas 
INCLUDES += $(hipconfig -C)

divergence_3d_benchmarks:
	${CXX} -c -O3 kernels_3d.cpp -o kernels_3d.o
	${FC} -c -O3 -fopenmp -ffree-line-length-512 divergence_3d.f90 ${INCLUDES} ${LIBS} -o divergence_3d.o
	${FC} -fopenmp -ffree-line-length-512 kernels_3d.o divergence_3d.o ${INCLUDES} ${LIBS} -o $@

divergence_2d_benchmarks:
	${CXX} -c -O3 kernels_2d.cpp -o kernels_2d.o
	${FC} -c -O3 -fopenmp -ffree-line-length-512 divergence_2d.f90 ${INCLUDES} ${LIBS} -o divergence_2d.o
	${FC} -fopenmp -ffree-line-length-512 kernels_2d.o divergence_2d.o ${INCLUDES} ${LIBS} -o $@

.PHONY: clean

clean:
	rm -f divergence_3d_benchmarks divergence_2d_benchmarks *.o *.mod
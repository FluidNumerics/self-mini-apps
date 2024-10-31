# SELF Mini-Apps

This repository contains mini-apps that can be used for experimenting with new implementations for the core algorithm's in the [Spectral Element Library in Fortran](https://github.com/fluidnumerics/self).

## Some benchmark results

### Divergence (2-D)

#### Fluid Numerics - "Noether" - AMD MI210
These results show the effective bandwidth (in GB/s) for the various 2-D divergence kernel implementations on Fluid Numerics' "Noether" platform. This system is equipped with 4x AMD MI210 GPUs and two AMD EPYC 7313 16-Core Processors, giving a 8 vCPU/GPU ratio. All benchmarks are shown for 1 MI210 GPU; for CPU-only kernels, we use 8 threads for our `do concurrent` loops. The environment consists of the Ubuntu 22.04 operating system, with gfortran 12.3.0, and ROCm 6.2.1 .

![Screenshot from 2024-10-31 12-47-10](https://github.com/user-attachments/assets/758de07c-374b-4afb-b88f-9aa759349cc3)

[See the interactive report](https://lookerstudio.google.com/embed/reporting/98277cb6-767d-41b5-b20c-af7eee9939e1/page/6xdIE)

### Divergence (3-D)

#### Fluid Numerics - "Noether" - AMD MI210
These results show the effective bandwidth (in GB/s) for the various 3-D divergence kernel implementations on Fluid Numerics' "Noether" platform. This system is equipped with 4x AMD MI210 GPUs and two AMD EPYC 7313 16-Core Processors, giving a 8 vCPU/GPU ratio. All benchmarks are shown for 1 MI210 GPU; for CPU-only kernels, we use 8 threads for our `do concurrent` loops. The environment consists of the Ubuntu 22.04 operating system, with gfortran 12.3.0, and ROCm 6.2.1![Screenshot from 2024-10-31 13-11-02](https://github.com/user-attachments/assets/2533dade-2a65-4457-bbf6-c7b3eb2ee72c)

[See the interactive report](https://lookerstudio.google.com/embed/reporting/98277cb6-767d-41b5-b20c-af7eee9939e1/page/p_ftc798jimd)

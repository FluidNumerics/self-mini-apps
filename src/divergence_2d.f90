! //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !
!
! Maintainers : support@fluidnumerics.com
! Official Repository : https://github.com/FluidNumerics/self/
!
! Copyright © 2024 Fluid Numerics LLC
!
! Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
!    the documentation and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
! HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
! LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
! THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
! THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !

module divergence_2d_kernels

use SELF_Constants
use SELF_Lagrange
use SELF_GPUBLAS
use iso_c_binding

implicit none

interface
  subroutine  divergence_2d_gpu(f,df,dmatrix,N,nel,nvar) bind(c,name="divergence_2d_gpu")
    use iso_c_binding
    implicit none
    type(c_ptr),value :: f,df,dmatrix
    integer(c_int),value :: N,nel,nvar
  endsubroutine divergence_2d_gpu
endinterface

interface
  subroutine  divergence_2d_naive_gpu(f,df,dmatrix,N,nel,nvar) bind(c,name="divergence_2d_naive_gpu")
    use iso_c_binding
    implicit none
    type(c_ptr),value :: f,df,dmatrix
    integer(c_int),value :: N,nel,nvar
  endsubroutine divergence_2d_naive_gpu
endinterface

interface
  subroutine  divergence_2d_sm_gpu(f,df,dmatrix,N,nel,nvar) bind(c,name="divergence_2d_sm_gpu")
    use iso_c_binding
    implicit none
    type(c_ptr),value :: f,df,dmatrix
    integer(c_int),value :: N,nel,nvar
  endsubroutine divergence_2d_sm_gpu
endinterface

interface
  subroutine  divergence_2d_naive_sm_gpu(f,df,dmatrix,N,nel,nvar) bind(c,name="divergence_2d_naive_sm_gpu")
    use iso_c_binding
    implicit none
    type(c_ptr),value :: f,df,dmatrix
    integer(c_int),value :: N,nel,nvar
  endsubroutine divergence_2d_naive_sm_gpu
endinterface

contains
subroutine divergence_doconcurrent(f,df,interp,nelem,nvar)
    implicit none
    type(Lagrange),intent(in) :: interp
    real(prec),intent(in) :: f(1:interp%N+1,1:interp%N+1,1:nelem,1:nvar,1:2)
    real(prec),intent(out) :: df(1:interp%N+1,1:interp%N+1,1:nelem,1:nvar)
    integer, intent(in) :: nelem,nvar
    ! Local
    integer    :: i,j,ii,iel,ivar
    real(prec) :: dfLoc

    do concurrent(i=1:interp%N+1,j=1:interp%N+1, &
                  iel=1:nelem,ivar=1:nvar)

      dfLoc = 0.0_prec
      do ii = 1,interp%N+1
        dfLoc = dfLoc+interp%dMatrix(ii,i)*f(ii,j,iel,ivar,1)
      enddo
      dF(i,j,iel,ivar) = dfLoc

    enddo

    do concurrent(i=1:interp%N+1,j=1:interp%N+1, &
                  iel=1:nelem,ivar=1:nvar)

      dfLoc = 0.0_prec
      do ii = 1,interp%N+1
        dfLoc = dfLoc+interp%dMatrix(ii,j)*f(i,ii,iel,ivar,2)
      enddo
      dF(i,j,iel,ivar) = dF(i,j,iel,ivar)+dfLoc

    enddo

  endsubroutine divergence_doconcurrent

  subroutine divergence_naive_doconcurrent(f,df,interp,nelem,nvar)
    implicit none
    type(Lagrange),intent(in) :: interp
    real(prec),intent(in) :: f(1:interp%N+1,1:interp%N+1,1:nelem,1:nvar,1:2)
    real(prec),intent(out) :: df(1:interp%N+1,1:interp%N+1,1:nelem,1:nvar)
    integer, intent(in) :: nelem,nvar
    ! Local
    integer    :: i,j,ii,iel,ivar
    real(prec) :: dfLoc

    do concurrent(i=1:interp%N+1,j=1:interp%N+1, &
                  iel=1:nelem,ivar=1:nvar)

      dfLoc = 0.0_prec
      do ii = 1,interp%N+1
        dfLoc = dfLoc+interp%dMatrix(ii,i)*f(ii,j,iel,ivar,1)+&
                      interp%dMatrix(ii,j)*f(i,ii,iel,ivar,2)

      enddo
      dF(i,j,iel,ivar) = dfLoc

    enddo

  endsubroutine divergence_naive_doconcurrent

  subroutine divergence_gpublas(f,df,interp,nelem,nvar,blas_handle)
    implicit none
    type(c_ptr),intent(in) :: f
    type(c_ptr),intent(inout) :: df
    type(Lagrange),intent(in) :: interp
    integer, intent(in) :: nelem,nvar
    type(c_ptr),intent(in) :: blas_handle
    !Local
    real(prec),pointer :: f_p(:,:,:,:,:)
    type(c_ptr) :: fc

    call c_f_pointer(f,f_p, &
                     [interp%N+1,interp%N+1,nelem,nvar,2])

    fc = c_loc(f_p(1,1,1,1,1))
    call self_blas_matrixop_dim1_2d(interp%dMatrix_gpu,fc,df, &
                                    interp%N,interp%N,nvar,nelem,blas_handle)

    fc = c_loc(f_p(1,1,1,1,2))
    call self_blas_matrixop_dim2_2d(interp%dMatrix_gpu,fc,df, &
                                    1.0_c_prec,interp%N,interp%N,nvar,nelem,blas_handle)


    f_p => null()

  endsubroutine divergence_gpublas

endmodule divergence_2d_kernels

program divergence_benchmarks

use divergence_2d_kernels
use SELF_Scalar_2d
use SELF_Vector_2d
use omp_lib

implicit none

integer, parameter :: N = 7
integer, parameter :: M = 13
integer, parameter :: nvar = 7
integer, parameter :: nelem = 5000
integer, parameter :: nrepeats = 100 ! How many times to call

type(Lagrange),target :: interp
type(Vector2d) :: f
type(Scalar2d) :: df
type(c_ptr) :: magma_queue
integer :: i
integer(kind=8) :: ndof
real(prec) :: est_flops, est_bytes
real(prec) :: t1, t2, wall_time, avg_wall_time


print*, ""
print*, "                            ░▒▓▓█████████▓▒▒░"
print*, "                         ░▓███████████████████▓░"
print*, "                       ▒████████▓▓▒▒▒▒▒▓▓████████▒"
print*, "                     ▓████▓▒░              ░░▓█████▒"
print*, "                   ▒████▒                      ░▒████░"
print*, "                  ▓██▓░                           ▒███▒"
print*, "                ░███░                               ▒██▓"
print*, "                ██▓                                  ░██▓"
print*, "               ██▓                                    ░██▒"
print*, "              ▒██                                      ░██░"
print*, "              ██░    ░▒▒░                      ░▒▒▒     ▓██"
print*, "             ▒██    ▒█████▒                   ▓█████░   ░██░"
print*, "             ██▓    ███████▓                 ████████    ██▒"
print*, "             ██▓    ████  ▒█▒               ▓█░ ░███▓    ██▓"
print*, "             ███    ░▓█▒   █▓               █▓   ▓█▓    ░██▓"
print*, "             ███▒          █▓               █▓          ▓██▓"
print*, "             ▓███         ░█░               ▓█         ░███▒"
print*, "             ░███▒        █▓                 █▓        ▓███"
print*, "              ▓███▒     ░█▓                  ░█▓░     ▓███▒"
print*, "               ▓███▓▒▒▓██▓                    ░▓██▒▒▓████▒"
print*, "                ░▓█████▒░                       ░▓█████▒░"
print*, ""
print*, "                ▒▒▒▒▒▒▒▒▒ ▒▒        ▒▒     ▒▒ ▒▒ ▒▒▒▒▒▒▒▒"
print*, "                ▒▒        ▒▒        ▒▒     ▒▒ ▒▒ ▒▒    `▒▒"
print*, "               ▒▒▒▒▒▒▒    ▒▒        ▒▒     ▒▒ ▒▒ ▒▒     ▒▒"
print*, "                ▒▒        ▒▒        ▒▒     ▒▒ ▒▒ ▒▒     ▒▒"
print*, "                ▒▒        ▒▒        ▒▒.   .▒▒ ▒▒ ▒▒    .▒▒"
print*, "                ▒▒        ▒▒▒▒▒▒▒▒▒ `▒▒▒▒▒▒▒' ▒▒ ▒▒▒▒▒▒▒▒"
print*, ""
print*, "▒▒▒▒▒▒▒▒  ▒▒     ▒▒ ▒▒▒▒▒▒.▒▒▒▒   ▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒. .▒▒▒▒▒▒▒"
print*, "▒▒    `▒▒ ▒▒     ▒▒ ▒▒  `▒▒  `▒▒  ▒▒         ▒▒    `▒▒ ▒▒ ▒▒'   `▒▒ ▒▒.     '"
print*, "▒▒     ▒▒ ▒▒     ▒▒ ▒▒   ▒▒   ▒▒ ▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒' ▒▒ ▒▒        `▒▒▒▒▒▒▒."
print*, "▒▒     ▒▒ ▒▒     ▒▒ ▒▒   ▒▒   ▒▒  ▒▒         ▒▒   `▒▒. ▒▒ ▒▒              `▒▒"
print*, "▒▒     ▒▒ ▒▒.   .▒▒ ▒▒   ▒▒   ▒▒  ▒▒         ▒▒     ▒▒ ▒▒ ▒▒.   .▒▒ ▒▒'   .▒▒"
print*, "▒▒     ▒▒ `▒▒▒▒▒▒▒' ▒▒   ▒▒   ▒▒  ▒▒▒▒▒▒▒▒▒  ▒▒     ▒▒ ▒▒  ▒▒▒▒▒▒▒'  ▒▒▒▒▒▒▒"
print*, ""
print*, ""


  ! The number of degrees of freedom
  ndof = (N+1)*(N+1)*nvar*nelem

  ! Estimated number of flops. For each degree of freedom, we perform three vector-vector products.
  ! The multiplication by two occurs since each component of the vector-vector product is a FMA
  est_flops = real(ndof*(N+1)*2*2,prec)/10.0_prec**9 ! in GFLOPs 

  ! For each vector-vector product, we read 
  !   > N+1 derivative matrix entries
  !   > N+1 vector values
  ! and we write
  !   > 1 scalar value
  est_bytes = real(ndof*( 2*(N+1) + 1 )*prec,prec)/10.0_prec**9 ! in GB

  print*, " =============================================================================== "
  print*, " (SELF) Divergence 2d Kernel Benchmark                     "
  print*, " =============================================================================== "
  print*, " N. Repeats          : ", nrepeats
  print*, " Control degree      : ", N
  print*, " No. Elements        : ", nElem
  print*, " No. Variables       : ", nVar
  print*, " Degrees of Freedom  : ", ndof
  print*, " Est. GFLOPs         : ", est_flops
  print*, " Est. Bytes (MB)     : ", est_bytes
  print*, " Est. FLOPs/Byte     : ", real(est_flops,prec)/real(est_bytes,prec)
  print*, " =============================================================================== "

  ! Create an interpolant
  call interp%Init(N=N, &
                   controlNodeType=GAUSS, &
                   M=M, &
                   targetNodeType=UNIFORM)

   ! Initialize vectors
   call f%Init(interp,nvar,nelem)
   
   call df%Init(interp,nvar,nelem)
   
   ! Set the source vector (on the control grid) to a non-zero constant
   f%interior = 1.0_prec

   call f%UpdateDevice()

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_doconcurrent(f%interior,df%interior,interp,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = t2-t1
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))

   print*, "   Divergence (do concurrent)"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_naive_doconcurrent(f%interior,df%interior,interp,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = t2-t1
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))

   print*, "   Divergence (do concurrent [naive])"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_2d_naive_gpu(f%interior_gpu,df%interior_gpu,interp%dMatrix_gpu,interp%N,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = t2-t1
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))


   print*, "   Divergence (hip kernel [naive])"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "



   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_gpublas(f%interior_gpu,df%interior_gpu,interp,nelem,nvar,f%blas_handle)
   enddo
   t2 = omp_get_wtime()

   wall_time = t2-t1
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))


   print*, "   Divergence (hipblas)"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_2d_gpu(f%interior_gpu,df%interior_gpu,interp%dMatrix_gpu,interp%N,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = t2-t1
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))
   call df%UpdateHost()

   if(maxval(abs(df%interior)) >= 10.0_prec**(-7)) then
     print*, "your kernel sucks!"
     print*, maxval(abs(df%interior))
   endif

   print*, "   Divergence (hip kernel)"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_2d_sm_gpu(f%interior_gpu,df%interior_gpu,interp%dMatrix_gpu,interp%N,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = (t2-t1)
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))
   call df%UpdateHost()

   if(maxval(abs(df%interior)) >= 10.0_prec**(-7)) then
     print*, "your kernel sucks!"
     print*, maxval(abs(df%interior))
   endif

   print*, "   Divergence (hip kernel [shared memory])"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "

   t1 = omp_get_wtime()
   do i = 1, nrepeats
    call divergence_2d_naive_sm_gpu(f%interior_gpu,df%interior_gpu,interp%dMatrix_gpu,interp%N,nelem,nvar)
   enddo
   t2 = omp_get_wtime()

   wall_time = (t2-t1)
   avg_wall_time = (t2-t1)/(real(nrepeats,prec))
   call df%UpdateHost()

   if(maxval(abs(df%interior)) >= 10.0_prec**(-7)) then
     print*, "your kernel sucks!"
     print*, maxval(abs(df%interior))
   endif

   print*, "   Divergence (hip kernel [naive+shared memory])"
   print*, " ------------------------------------------------------------------------------- "
   print*, " Total Wall Time (s)   : ", wall_time
   print*, " Avg. Wall Time (s)    : ", avg_wall_time
   print*, " Est. GFLOPs/sec       : ", real(est_flops,prec)/avg_wall_time
   print*, " Est. Bandwidth (GB/s) : ", real(est_bytes,prec)/avg_wall_time
   print*, " =============================================================================== "



   call f%free()
   call df%free()
   call interp%free()

endprogram divergence_benchmarks
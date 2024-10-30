
#include <hip/hip_runtime.h>


__global__ void __launch_bounds__(256) divergence_2d_dim1_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x + blockIdx.x*blockDim.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.y;
        uint32_t ivar = blockIdx.z;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*i]*f[ii+(N+1)*(j+(N+1)*(iel + nel*ivar))];
        }
        df[idof + nq*(iel + nel*ivar)] = dfloc;
    }

}

__global__ void __launch_bounds__(256) divergence_2d_dim2_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x + blockIdx.x*blockDim.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.y;
        uint32_t ivar = blockIdx.z;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        double dfloc = 0.0; 
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*j]*f[i+(N+1)*(ii+(N+1)*(iel + nel*(ivar + nvar)))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}


extern "C"
{
  void divergence_2d_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1);
    int threads_per_block = 64;
    int nblocks_x = nq/threads_per_block;

    divergence_2d_dim1_gpukernel<<<dim3(nblocks_x,nel,nvar), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    divergence_2d_dim2_gpukernel<<<dim3(nblocks_x,nel,nvar), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    hipDeviceSynchronize();
  }

}

__global__ void __launch_bounds__(256) divergence_2d_naive_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x + blockIdx.x*blockDim.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.y;
        uint32_t ivar = blockIdx.z;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*i]*f[ii+(N+1)*(j+(N+1)*(iel + nel*(ivar)))]+
                     dmatrix[ii+(N+1)*j]*f[i+(N+1)*(ii+(N+1)*(iel + nel*(ivar + nvar)))]
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

extern "C"
{
  void divergence_2d_naive_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1);
    int threads_per_block = 64;
    int nblocks_x = nq/threads_per_block;

    divergence_2d_naive_gpukernel<<<dim3(nblocks_x,nel,nvar), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    hipDeviceSynchronize();
  }

}

__global__ void __launch_bounds__(512) divergence_2d_dim1_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x; // + blockIdx.x*blockDim.x;
    //if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        // assuming N+1 = 8 
        __shared__ double floc[64];
        floc[i+(N+1)*j] = f[i+(N+1)*(j+(N+1)*(iel + nel*ivar))];
        __syncthreads();

        double dfloc = 0.0;
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*i]*floc[ii+(N+1)*j];
        }
        df[idof + nq*(iel + nel*ivar)] = dfloc;
    //}

}

__global__ void __launch_bounds__(512) divergence_2d_dim2_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x; // + blockIdx.x*blockDim.x;
    //if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        __shared__ double floc[64];
        floc[i+(N+1)*(j)] = f[i+(N+1)*(j+(N+1)*(iel + nel*(ivar + nvar)))];
        __syncthreads();

        double dfloc = 0.0; 
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*j]*floc[i+(N+1)*ii]; 
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
   // }

}

extern "C"
{
  void divergence_2d_sm_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1);
    int threads_per_block = 64;
    int nblocks_x = nq/threads_per_block;

    divergence_2d_dim1_sm_gpukernel<<<dim3(nel,nvar,1), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    divergence_2d_dim2_sm_gpukernel<<<dim3(nel,nvar,1), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    hipDeviceSynchronize();
  }

}

__global__ void __launch_bounds__(512) divergence_2d_naive_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x + blockIdx.x*blockDim.x;
    //if( idof < nq ){
        
        uint32_t iel = blockIdx.y;
        uint32_t ivar = blockIdx.z;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);

        __shared__ double f1[64];
        __shared__ double f2[64];
        __shared__ double dmloc[64];
        f1[i+(N+1)*(j)] = f[i+(N+1)*(j+(N+1)*(iel + nel*(ivar)))];
        f2[i+(N+1)*(j)] = f[i+(N+1)*(j+(N+1)*(iel + nel*(ivar + nvar)))];
        if( k == 0 ){
            dmloc[i+(N+1)*j] = dmatrix[i+(N+1)*j];
        }
        __syncthreads();

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmloc[ii+(N+1)*i]*f1[ii+(N+1)*(j)]+
                     dmloc[ii+(N+1)*j]*f2[i+(N+1)*(ii)]+
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    //}

}

extern "C"
{
  void divergence_2d_naive_sm_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1)*(N+1);
    int threads_per_block = 512;
   // int nblocks_x = nq/threads_per_block;

    divergence_2d_naive_sm_gpukernel<<<dim3(1,nel,nvar), dim3(threads_per_block,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    hipDeviceSynchronize();
  }

}


#include <hip/hip_runtime.h>


__global__ void __launch_bounds__(512) divergence_3d_dim1_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*i]*f[ii+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*ivar)))];
        }
        df[idof + nq*(iel + nel*ivar)] = dfloc;
    }

}

__global__ void __launch_bounds__(512) divergence_3d_dim2_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        double dfloc = 0.0; 
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*j]*f[i+(N+1)*(ii+(N+1)*(k + (N+1)*(iel + nel*(ivar + nvar))))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

__global__ void __launch_bounds__(512) divergence_3d_dim3_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*k]*f[i+(N+1)*(j+(N+1)*(ii + (N+1)*(iel + nel*(ivar + 2*nvar))))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

extern "C"
{
  void divergence_3d_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1)*(N+1);
    if( N < 4 ){
        divergence_3d_dim1_gpukernel<<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim2_gpukernel<<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim3_gpukernel<<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);   
    } else if( N >= 4 && N < 8 ){
        divergence_3d_dim1_gpukernel<<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim2_gpukernel<<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim3_gpukernel<<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);    
    }
    hipDeviceSynchronize();
  }

}

__global__ void __launch_bounds__(512) divergence_3d_naive_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmatrix[ii+(N+1)*i]*f[ii+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar))))]+
                     dmatrix[ii+(N+1)*j]*f[i+(N+1)*(ii+(N+1)*(k + (N+1)*(iel + nel*(ivar + nvar))))]+
                     dmatrix[ii+(N+1)*k]*f[i+(N+1)*(j+(N+1)*(ii + (N+1)*(iel + nel*(ivar + 2*nvar))))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

extern "C"
{
  void divergence_3d_naive_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1)*(N+1);

    if( N < 4 ){
        divergence_3d_naive_gpukernel<<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);

    } else if( N >= 4 && N < 8 ){
        divergence_3d_naive_gpukernel<<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    }
    hipDeviceSynchronize();
  }

}

template<int blockSize, int matSize>
__global__ void __launch_bounds__(512) divergence_3d_dim1_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));


        __shared__ double floc[blockSize];
        __shared__ double dmloc[matSize];
        floc[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*ivar)))];
        if( k == 0 ){
            dmloc[i+(N+1)*j] = dmatrix[i+(N+1)*j];
        }
        __syncthreads();

        double dfloc = 0.0;
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmloc[ii+(N+1)*i]*floc[ii+(N+1)*(j+(N+1)*k)];
        }
        df[idof + nq*(iel + nel*ivar)] = dfloc;
    }

}

template<int blockSize, int matSize>
__global__ void __launch_bounds__(512) divergence_3d_dim2_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        __shared__ double floc[blockSize];
        __shared__ double dmloc[matSize];
        floc[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar + nvar))))];
        if( k == 0 ){
            dmloc[i+(N+1)*j] = dmatrix[i+(N+1)*j];
        }
        __syncthreads();

        double dfloc = 0.0; 
        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmloc[ii+(N+1)*j]*floc[i+(N+1)*(ii+(N+1)*k)]; //f[i+(N+1)*(ii+(N+1)*(k + (N+1)*(iel + nel*(ivar + nvar))))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
   }

}

template<int blockSize, int matSize>
__global__ void __launch_bounds__(512) divergence_3d_dim3_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
   if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        __shared__ double floc[blockSize];
        __shared__ double dmloc[matSize];
        floc[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar + 2*nvar))))];
        if( k == 0 ){
            dmloc[i+(N+1)*j] = dmatrix[i+(N+1)*j];
        }
        __syncthreads();

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmloc[ii+(N+1)*k]*floc[i+(N+1)*(j+(N+1)*ii)];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

extern "C"
{
  void divergence_3d_sm_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1)*(N+1);

    if( N < 4 ){
        divergence_3d_dim1_sm_gpukernel<64,16><<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim2_sm_gpukernel<64,16><<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim3_sm_gpukernel<64,16><<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);    
    } else if( N >= 4 && N < 8 ){
        divergence_3d_dim1_sm_gpukernel<512,64><<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim2_sm_gpukernel<512,64><<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
        divergence_3d_dim3_sm_gpukernel<512,64><<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);    
    }

    hipDeviceSynchronize();
  }

}

template<int blockSize, int matSize>
__global__ void __launch_bounds__(512) divergence_3d_naive_sm_gpukernel(double *f, double *df, double *dmatrix, int nq, int N, int nel, int nvar){

    uint32_t idof = threadIdx.x;
    if( idof < nq ){
        
        uint32_t iel = blockIdx.x;
        uint32_t ivar = blockIdx.y;
        uint32_t i = idof % (N+1);
        uint32_t j = (idof/(N+1)) % (N+1);
        uint32_t k = (idof/(N+1)/(N+1));

        __shared__ double f1[blockSize];
        __shared__ double f2[blockSize];
        __shared__ double f3[blockSize];
        __shared__ double dmloc[matSize];
        f1[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar))))];
        f2[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar + nvar))))];
        f3[i+(N+1)*(j+(N+1)*k)] = f[i+(N+1)*(j+(N+1)*(k + (N+1)*(iel + nel*(ivar + 2*nvar))))];
        if( k == 0 ){
            dmloc[i+(N+1)*j] = dmatrix[i+(N+1)*j];
        }
        __syncthreads();

        double dfloc = 0.0;

        for(int ii = 0; ii<N+1; ii++){
            dfloc += dmloc[ii+(N+1)*i]*f1[ii+(N+1)*(j+(N+1)*(k))]+
                     dmloc[ii+(N+1)*j]*f2[i+(N+1)*(ii+(N+1)*(k))]+
                     dmloc[ii+(N+1)*k]*f3[i+(N+1)*(j+(N+1)*(ii))];
        }
        df[idof + nq*(iel + nel*ivar)] += dfloc;
    }

}

extern "C"
{
  void divergence_3d_naive_sm_gpu(double *f, double *df, double *dmatrix, int N, int nel, int nvar){
    int nq = (N+1)*(N+1)*(N+1);
    if( N < 4 ){
        divergence_3d_naive_sm_gpukernel<64,16><<<dim3(nel,nvar,1), dim3(64,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);

    } else if( N >= 4 && N < 8 ){
        divergence_3d_naive_sm_gpukernel<512,64><<<dim3(nel,nvar,1), dim3(512,1,1), 0, 0>>>(f,df,dmatrix,nq,N,nel,nvar);
    }

    hipDeviceSynchronize();
  }

}

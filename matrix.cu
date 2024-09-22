#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

const __device__ int threadnx = 32;
__global__ void trans01_matrix_mul_1(float* x, float * y, float* z, int m, int n, int l);
__global__ void trans01_matrix_mul_2(float* x, float * y, float* z, int m, int n, int l);
__global__ void hello_from_gpu()
{
    printf("Hello World from the the GPU\n");
}

__global__ void matrix_mul(float* x, float * y, float* z, int m, int n, int l)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = idx / m;
  const int col = idx % m;

  if(row < m && col < n) {
    for(int i = 0; i < l; i++) {
      z[row*n+ col] += x[row*l + i] * y[i*n + col];

    }
    // if(row*n + col < 10){
    //   printf("z[%d]:%f\t\n",row*n + col,z[row*n+ col]);
    // }
  }
}

__global__ void matrix_mul_1(float* x, float * y, float* z, int m, int n, int l)
{
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  // for(; bidx < m; bidx += gridDim.x)
  {
    for(;tidx < n; tidx += blockDim.x) {
      for(int i = 0; i < l; i++) {
        z[bidx*n + tidx] += x[bidx*l + i] * y[i*n + tidx];
      }
    }
  }
}

__global__ void matrix_mul_2(float* x, float * y, float* z, int m, int n, int l)
{
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  extern __shared__ float data[];
  for(int i = tidx; i < l; i += blockDim.x) {
    data[i] = x[bidx*l + i];
  }

  // 注意调用这个函数保证该 block 里面所有的线程同步， 
  // 因为该 block 里面所有的线程需要协同工作，一起将 m*l 矩阵中的第 bidx 行的元素写入 data 中。
  __syncthreads();

  // for(; bidx < m; bidx += gridDim.x)
  {
    for(;tidx < n; tidx += blockDim.x) {
      for(int i = 0; i < l; i++) {
        z[bidx*n + tidx] += data[i] * y[i*n + tidx];
      }
    }
  }
}

__global__ void trans01_matrix_mul(float* x, float * y, float* z, int m, int n, int l){
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  int block_size = blockDim.x * blockDim.y;
  extern __shared__ float data[];
  for(int i = tid; i < l; i += block_size){
    data[i] = x[bid * l + i];
  }
  __syncthreads();
  // for(; bidx < m; bidx += gridDim.x * gridDim.y)
  {
    for(;tid < n; tid += block_size){
      for(int i = 0;i < l;i++){
        z[bid * n + tid] += data[i] * y[tid * l + i];
      }
    }
  }
}

__global__ void general_matrix_mul_0(float* x,bool transX, float * y,bool transY, float* z, int m, int n, int l){
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程对应的行索引
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程对应的列索引

  if (row < m && col < n) {
      float sum = 0.0f;

      for (int i = 0; i < l; ++i) {
          float x_value = transX ? x[i * m + row] : x[row * l + i];  
          float y_value = transY ? y[col * l + i] : y[i * n + col];  

          sum += x_value * y_value;
      }

      z[row * n + col] += sum;  
  }
}

void trans01_matrix_mul_cuda(float* x, float * y, float* z, int m, int n, int l){
  if(x == NULL || y == NULL || z == NULL)
    return ;
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, m*l*sizeof(float));
  cudaMalloc((void**)&d_y, n*l*sizeof(float));
  cudaMalloc((void**)&d_z, m*n*sizeof(float));

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, m*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, n*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_z, (void*)z, m*n*sizeof(float), cudaMemcpyHostToDevice);

  // dim3 blocks(m);
  // dim3 threads(32,32);
  // trans01_matrix_mul <<<blocks, threads, sizeof(float)*l>>>(d_x, d_y, d_z, m, n, l);
  // 将device得到的结果拷贝到host

  int bx = (m + threadnx - 1) / threadnx;
  int by = (n + threadnx - 1) / threadnx;
  dim3 blocks(bx, by);
  dim3 threads(threadnx, threadnx);
  // trans01_matrix_mul_1<<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);
  trans01_matrix_mul_2<<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);

  cudaMemcpy((void*)z, (void*)d_z, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

}

void general_mul_cuda(float* x,bool transX, float * y,bool transY, float* z, int m, int n, int l){
  if(x == NULL || y == NULL || z == NULL)
    return ;
  float *d_x, *d_y, *d_z;
  
  cudaMalloc((void**)&d_x, m*l*sizeof(float));
  cudaMalloc((void**)&d_y, l*n*sizeof(float));
  cudaMalloc((void**)&d_z, m*n*sizeof(float));

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, m*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, l*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_z, (void*)z, m*n*sizeof(float), cudaMemcpyHostToDevice);
  // 定义kernel的执行配置
  // dim3 threads(1024);
  // dim3 blocks(m*n-1024+1/1024);
  // matrix_mul <<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);

  // dim3 blocks(m);
  // dim3 threads(32,32);
  // matrix_mul_1 <<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  general_matrix_mul_0 <<<threadsPerBlock, numBlocks>>>(d_x, transX,d_y,transY, d_z, m, n, l);

  // 将device得到的结果拷贝到host

  cudaMemcpy((void*)z, (void*)d_z, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
}


__global__ void matrix_mul_3(float* x, float * y, float* z, int m, int n, int l){
  __shared__ float matA[threadnx][threadnx];
  __shared__ float matB[threadnx][threadnx];
  const int tidr = threadIdx.x;
  const int tidc = threadIdx.y;
  const int bidr = blockIdx.x * threadnx;
  const int bidc = blockIdx.y * threadnx;
  double results = 0.0;

  for(int j = 0; j < l; j += threadnx) {
    if(bidr + tidr < m && tidc + j < l) {
      matA[tidr][tidc] = x[(tidr + bidr) * l + tidc + j];
    } else {
      matA[tidr][tidc] = 0.0;
    }

    if(tidr + j < l && bidc + tidc < n) {
      matB[tidr][tidc] = y[(tidr + j) * n + bidc + tidc];
    } else {
      matB[tidr][tidc] = 0.0;
    }

    __syncthreads();
    
    for(int i = 0; i < threadnx; i++) {
      results += matA[tidr][i] * matB[i][tidc];
    }

    __syncthreads();
  }

  if(tidr + bidr < m && tidc + bidc < n) {
    z[(tidr + bidr)*n + tidc + bidc] += results;
  }
}

__global__ void trans01_matrix_mul_1(float* x, float * y, float* z, int m, int n, int l){
  __shared__ float matA[threadnx][threadnx];
  __shared__ float matB[threadnx][threadnx];
  const int tidr = threadIdx.x;
  const int tidc = threadIdx.y;
  const int bidr = blockIdx.x * threadnx;
  const int bidc = blockIdx.y * threadnx;
  double results = 0.0;
  
  for(int j = 0; j < l; j += threadnx){
    if(tidr + bidr < m && tidc + j < l){
      matA[tidr][tidc] = x[(tidr + bidr) * l + tidc + j]; 
    }else{
      matA[tidr][tidc] = 0;
    }

    if(tidc + bidc < n && tidc + j < l){
      matB[tidr][tidc] = y[(tidc + bidc) * l + tidc + j];
    }else{
      matB[tidr][tidr] = 0;
    }
    __syncthreads();
    // 
    // 可以进行优化 先并行乘法，后加法
    for(int i = 0; i < threadnx ; i++){
      results += matA[tidr][i] * matB[tidr][i];
    }
    __syncthreads();

  }
  if(tidr + bidr < m &&  tidc + bidc < n){
    z[(tidr + bidr) * n + tidc + bidc] += results;
  }
}


__global__ void trans01_matrix_mul_2(float* x, float * y, float* z, int m, int n, int l){
  __shared__ float matA[threadnx][threadnx];
  __shared__ float matB[threadnx][threadnx];
  __shared__ float matT[threadnx][threadnx];

  const int tidr = threadIdx.x;
  const int tidc = threadIdx.y;
  const int bidr = blockIdx.x * threadnx;
  const int bidc = blockIdx.y * threadnx;
  double results = 0.0;
  for(int j = 0; j < l; j += threadnx){
    if(tidr + bidr < m && tidc + j < l){
      matA[tidr][tidc] = x[(tidr + bidr) * l + tidc + j]; 
    }else{
      matA[tidr][tidc] = 0;
    }

    if(tidc + bidc < n && tidc + j < l){
      matB[tidr][tidc] = y[(tidc + bidc) * l + tidc + j];
    }else{
      matB[tidr][tidr] = 0;
    }
    __syncthreads();
    // 
    // 是否可以进行优化 先并行乘法，后加法
    // for(int i = 0; i < threadnx ; i++){
    //   results += matA[tidr][i] * matB[tidr][i];
    // }
    // if()
    // matT[tidr][tidc] = matA[tidr][tidc] * matB[tidr][tidc];
    // TODO 错误的并行 why？

    for(int i = 0; i < threadnx ; i++){
      results += matT[tidr][i];
    }
    __syncthreads();

  }

  if(tidr + bidr < m &&  tidc + bidc < n){
    z[(tidr + bidr) * n + tidc + bidc] += results;
  }
}


void matrix_mul_cuda(float* x, float * y, float* z, int m, int n, int l){
  if(x == NULL || y == NULL || z == NULL)
    return ;
  float *d_x, *d_y, *d_z;
  
  cudaMalloc((void**)&d_x, m*l*sizeof(float));
  cudaMalloc((void**)&d_y, l*n*sizeof(float));
  cudaMalloc((void**)&d_z, m*n*sizeof(float));

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, m*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, l*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_z, (void*)z, m*n*sizeof(float), cudaMemcpyHostToDevice);
  // 定义kernel的执行配置
  // dim3 threads(1024);
  // dim3 blocks(m*n-1024+1/1024);
  // matrix_mul <<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);

  dim3 blocks(m);
  dim3 threads(1024);
  // matrix_mul_1 <<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);
  matrix_mul_2 <<<blocks, threads, sizeof(float)*l>>>(d_x, d_y, d_z, m, n, l);

  // int bx = (m + threadnx - 1) / threadnx;
  // int by = (n + threadnx - 1) / threadnx;
  // dim3 blocks(bx, by);
  // dim3 threads(threadnx, threadnx);

  // matrix_mul_3 <<<blocks, threads>>>(d_x, d_y, d_z, m, n, l);


  // 将device得到的结果拷贝到host
  cudaMemcpy((void*)z, (void*)d_z, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // std::cout<<"cuda : ";
  // for(int i = 0; i < 10; i++) {
  //   std::cout << z[i] << " ";
  // }
  // std::cout<<""<<std::endl;
}




/*int main()
{
  int M = 2048;
  int L = 1024;
  int N = 512;

  // hello_from_gpu<<<4, 4>>>();
  // cudaDeviceSynchronize();
  // 申请host内存
  float *x = NULL;
  float *y = NULL;
  float *z = NULL;
  x = (float*)malloc(M*L*sizeof(float));
  y = (float*)malloc(L*N*sizeof(float));
  z = (float*)malloc(M*N*sizeof(float));

  if(x == NULL || y == NULL || z == NULL)
    return 0;
  
  // 初始化数据
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < L; ++j) {
      x[i*L + j] = 1.1;
    }
  }
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < N; ++j) {
      y[i*N + j] = 1.1;
    }
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      z[i*N + j] = 0;
    }
  }

  // 申请device内存
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, M*L*sizeof(float));
  cudaMalloc((void**)&d_y, L*N*sizeof(float));
  cudaMalloc((void**)&d_z, M*N*sizeof(float));

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, M*L*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, L*N*sizeof(float), cudaMemcpyHostToDevice);

  // 定义kernel的执行配置
  dim3 threads(1024);
  dim3 blocks(M*N-1024+1/1024);
  matrix_mul <<<blocks, threads>>>(d_x, d_y, d_z, M, N, L);

  // 将device得到的结果拷贝到host
  cudaMemcpy((void*)z, (void*)d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  // 输出前10个数值
  for(int i = 0; i < 10; i++) {
    std::cout << z[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Done!" << std::endl;

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // 释放host内存
  free(x);
  free(y);
  free(z);

  return 0;
}
*/
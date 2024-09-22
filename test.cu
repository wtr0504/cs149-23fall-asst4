#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16  // 定义线程块的大小

// 核函数：使用棋盘阵列乘法计算矩阵 A 和 B_t 的乘积
__global__ void matrixMulWithTranspose(float* A, float* B, float* C, int m, int n, int l) {
    // 定义共享内存，用于存储块 A 和 B
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    // 计算当前线程对应的行和列
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;  // 用于存储最终的乘积结果

    // 以块为单位进行循环，遍历矩阵 A 和 B_t 的所有小块
    for (int k = 0; k < (l + TILE_WIDTH - 1) / TILE_WIDTH; k++) {
        // 每个线程从全局内存加载 A 和 B_t 的一块数据到共享内存
        if (row < m && (k * TILE_WIDTH + threadIdx.x) < l) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * l + k * TILE_WIDTH + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && (k * TILE_WIDTH + threadIdx.y) < l) {
            shared_B[threadIdx.y][threadIdx.x] = B[col * l + k * TILE_WIDTH + threadIdx.y];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步线程，确保当前块数据已加载完毕
        __syncthreads();

        // 每个线程计算对应的乘积，并将结果累加到 value
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        }

        // 再次同步，确保所有线程都完成了当前块的计算
        __syncthreads();
    }

    // 将计算结果写入到结果矩阵 C 中
    if (row < m && col < n) {
        C[row * n + col] = value;
    }
}

// 主机函数：调用 CUDA 核函数
void matrixMultiplyWithTranspose(float* A, float* B, float* C, int m, int n, int l) {
    float *d_A, *d_B, *d_C;

    // 为 A, B, C 矩阵分配设备内存
    cudaMalloc((void**)&d_A, m * l * sizeof(float));
    cudaMalloc((void**)&d_B, n * l * sizeof(float));  // B_t 的维度是 n x l
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // 将 A, B 数据从主机拷贝到设备
    cudaMemcpy(d_A, A, m * l * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * l * sizeof(float), cudaMemcpyHostToDevice);  // 注意 B_t 实际上是 B 的转置

    // 定义线程块和网格大小
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // 调用 CUDA 核函数
    matrixMulWithTranspose<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, l);

    // 将结果矩阵 C 从设备拷贝回主机
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int m = 4, n = 4, l = 4;

    // 定义矩阵 A 和 B_t
    float A[] = {1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 16};
    float B[] = {1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 16};

    // 定义结果矩阵 C
    float C[16];

    // 调用矩阵乘法函数
    matrixMultiplyWithTranspose(A, B, C, m, n, l);

    // 输出结果矩阵 C
    printf("Result matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}

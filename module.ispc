#define TILE_SIZE 16
export void MM2d_v(uniform float A[], uniform bool transA,
                   uniform float B[], uniform bool transB,
                   uniform float C[],
                   uniform int M, uniform int N, uniform int P) {
    /*
     * Credit to https://github.com/google/gemmlowp/blob/master/test/test.cc#L35 and
     * https://github.com/ispc/ispc/blob/main/examples/cpu/sgemm/SGEMM_kernels.ispc#L292
     */

    uniform unsigned int a_i_stride, a_k_stride, b_j_stride, b_k_stride;
    uniform float oneAVal;

    if (transA) {
        a_i_stride = 1;
        a_k_stride = M;
    } else {
        a_i_stride = P;
        a_k_stride = 1;
    }

    if (transB) {
        b_j_stride = P;
        b_k_stride = 1;
    } else {
        b_j_stride = 1;
        b_k_stride = N;
    }

    for (uniform unsigned int k = 0; k < P; k++) {
        for (uniform unsigned int j = 0; j < N; j+=TILE_SIZE) {        
            foreach (ji = 0 ... TILE_SIZE) {
                varying float matB = B[(j + ji) * b_j_stride + k * b_k_stride];
                for (uniform unsigned int i = 0; i < M; i++) {
                    oneAVal = A[i * a_i_stride + k * a_k_stride];
                    C[i * N + j + ji] += oneAVal * matB;
                }
            }
        }
    }
}

export uniform float rowSum_v(uniform float A[], uniform int N) {
    float psum = 0.0f;
    foreach	(i = 0 ... N) {
        psum += A[i];
    }
    return reduce_add(psum);
}
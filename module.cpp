#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include<numeric>
// Uncomment for ISPC
#include "module_ispc.h"
using namespace ispc;

void matrix_mul_cuda(float* x, float * y, float* z, int m, int n, int l);
void trans01_matrix_mul_cuda(float* x, float * y, float* z, int m, int n, int l);
void general_mul_cuda(float* x,bool transX, float * y,bool transY, float* z, int m, int n, int l);
// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x *(sizeX) *(sizeY) * (sizeZ) + y*sizeY*sizeZ + z * sizeZ+ b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[ x *(sizeX) *(sizeY) * (sizeZ) + y*sizeY*sizeZ + z * sizeZ+ b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
        std::cout<< "fuck 0"<<std::endl;

        for (int b = 0; b < B; b++) {
             //loop over Heads
            for (int h = 0; h < H; h++) {
                 //loop over Sequence Length
                for (int i = 0; i < N; i++) {
                    for(int j = 0; j < N;j++){
                        float tmp = 0;
                        for (int a = 0; a < d; a++) {
                            float val1 = fourDimRead(Q, b, h, i, a, H, N, d);
                            float val2 = fourDimRead(K, b, h, j, a, H, N, d);
                            tmp += val1 * val2;
                        }
                        twoDimWrite(QK_t, i, j, N, tmp);
                    }
                     //loop over Embedding Dimensionality
                }
                for(int i = 0; i < N ; i++){
                    float tmp = 0.0;
                    for(int j = 0; j < N;j++){
                        float val = twoDimRead(QK_t,i,j,N);
                        tmp += exp(val);
                    }
                    for(int j = 0; j < N; j ++){
                        float val = twoDimRead(QK_t,i,j,N);
                        float res = (float)exp(val)/tmp;
                        twoDimWrite(QK_t,i,j,N,res);
                    }
                }

                for(int i = 0; i < N; i++){
				    for(int k = 0; k < N; k++){
					    for(int j = 0; j < d; j++){
                            float val = fourDimRead(O, b, h, i, j, H, N, d);
                            val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                            fourDimWrite(O, b, h, i, j, H, N, d, val);
                        }
				    }
			    }

                // for(int i = 0;i < N;i++){
                //     for(int j = 0;j < d;j++){
                //         float tmp = 0.0;
                //         for(int a = 0; a < N; a++){
                //             float val1 = twoDimRead(QK_t,i,a,N);
                //             float val2 = fourDimRead(V,b,h,a,j,H,N,d);
                //             tmp += val1* val2;
                //         }
                //         fourDimWrite(O,b,h,i,j,H,N,d,tmp);
                //     }
                // }

            }
        }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //

    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    // const int L = 2;
    // int tile_size = N / L;
    // int x_left = std::min(tile_size, )
    

    for (int b = 0; b < B; b++) {

    //loop over Heads
        for (int h = 0; h < H; h++) {
            // std::fill(QK_t.begin(), QK_t.end(), 0);
        //loop over Sequence Length
            int B_size = 16;
            int H_size = 64;
            for(int is = 0; is < N;is += H_size){
                for(int js = 0; js < N; js += B_size){
                    for(int as = 0;as < d;as += B_size){
                        for(int i = is; i < std::min(is + H_size,N);i++){
                            for(int j = js; j < std::min(js + B_size,N);j++){
                                float tmp = twoDimRead(QK_t,i,j,N);
                                for(int a = as; a < std::min(as + B_size,d);a++){
                                    tmp += fourDimRead(Q,b,h,i,a,H,N,d) * fourDimRead(K,b,h,j,a,H,N,d);
                                }
                                twoDimWrite(QK_t,i,j,N,tmp);
                            }
                        }
                    }
                }
            }

            // for(int k = 0; k < L ;k++){
            //     for(int i = tile_size * k ; i < tile_size * (k+1);i ++){
            //         for(int j = tile_size * k; j < tile_size * (k+1);j++){

            //             float tmp = twoDimRead(QK_t,i,j,N);
            //             for(int a = tile_size * k; a < tile_size * (k+1);a++){
            //                 tmp += fourDimRead(Q,b,h,i,a,H,N,d) * fourDimRead(K,b,h,j,a,H,N,d);
            //             }
            //             twoDimWrite(QK_t,i,j,N,tmp);
            //         }
            //     }
            // }
            // int left_x = d % tile_size;
            // int left_y = N % tile_size;

            // for(int i = N - left_y; i < N;i++){
            //     for(int j = N - left_y;j < N;j++){
            //         float tmp = twoDimRead(QK_t,i,j,N);
            //         for (int a = 0; a < d; a++) {
            //             float val1 = fourDimRead(Q, b, h, i, a, H, N, d);
            //             float val2 = fourDimRead(K, b, h, j, a, H, N, d);
            //             tmp += val1 * val2;
            //         }
            //         twoDimWrite(QK_t, i, j, N, tmp);
            //     }
            // }
            // for(int i = 0; i < N - left_y;i++){
            //     for(int j = 0;j < N - left_y;j++){
            //         float tmp = twoDimRead(QK_t,i,j,N);
            //         for(int a = d - left_x;a < d;a++){
            //             tmp += fourDimRead(Q,b,h,i,a,H,N,d) * fourDimRead(K,b,h,j,a,H,N,d);
            //         }
            //         twoDimWrite(QK_t,i,j,N,tmp);
            //     }
            // }


            for(int i = 0; i < N ; i++){
                    float tmp = 0.0;
                    for(int j = 0; j < N;j++){
                        float val = twoDimRead(QK_t,i,j,N);
                        tmp += exp(val);
                    }
                    for(int j = 0; j < N; j ++){
                        float val = twoDimRead(QK_t,i,j,N);
                        float res = (float)exp(val)/tmp;
                        twoDimWrite(QK_t,i,j,N,res);
                    }
                }

                std::cout<<"fuck"<<std::endl;

                for(int is = 0; is < N; is += H_size){
                    for(int js = 0; js < d; js += B_size){
                        for(int ks = 0; ks < N; ks += B_size){
                            for(int i = is; i < std::min(is+H_size, N); i++){
                                for(int k = ks; k < std::min(ks+B_size ,N); k++){
                                    for(int j = js; j < std::min(js+B_size, d); j++){
                                        float val = fourDimRead(O, b, h, i, j, H, N, d);
                                        val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                                        fourDimWrite(O, b, h, i, j, H, N, d, val);
                                    }
                                    
                                }
                            }
                        }
                    }
                }



        }
    }



    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                float sft = 0.0;

                for(int j = 0; j < N;j++){
                    float tmp = 0;
                    for (int a = 0; a < d; a++) {
                        float val1 = fourDimRead(Q, b, h, i, a, H, N, d);
                        float val2 = fourDimRead(K, b, h, j, a, H, N, d);
                        tmp += val1 * val2;
                    }
                    ORow[j] = exp(tmp);
                    sft += exp(tmp);

                }

                for(int j = 0; j < N; j++){
                    ORow[j] = ORow[j]/sft;
                }

                for(int a = 0; a < N; a++){
                    for(int j = 0; j < d; j++){
                        float res = fourDimRead(O,b,h,i,j,H,N,d);
                        res += ORow[a] * fourDimRead(V,b,h,a,j,H,N,d);
                        fourDimWrite(O, b, h, i, j, H, N, d, res);
                    }
                }

            }
	    }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

void MM2d(std::vector<float> &A, bool transA, std::vector<float> &B, bool transB, 
          std::vector<float> &C, int M, int N, int P) {
    // credit to https://github.com/google/gemmlowp/blob/master/test/test.cc#L35
    int a_i_stride, a_k_stride;
    if (transA) {
        a_i_stride = 1;
        a_k_stride = M;
    } else {
        a_i_stride = P;
        a_k_stride = 1;
    }
    int b_j_stride, b_k_stride;
    if (transB) {
        b_j_stride = P;
        b_k_stride = 1;
    } else {
        b_j_stride = 1;
        b_k_stride = N;
    }
    
    for (int k = 0; k < P; k++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                const int a_index = i * a_i_stride + k * a_k_stride;
                const int b_index = j * b_j_stride + k * b_k_stride;
                float val = twoDimRead(C, i, j, N) + A[a_index] * B[b_index];
                twoDimWrite(C, i, j, N, val);
            }
    // std::cout<<"cpu : ";
    // int j = 0;
    // for(int i = 0; i < 10; i++) {
    //     std::cout << twoDimRead(C,j,i,N) << " ";
    // }
    // std::cout<<""<<std::endl;
}
// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //
/*
torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    int Tr = N/Br;
    int Tc = N/Bc;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for(int j = 0 ; j < Tr; j ++){
                for(int a = Bc * j,b=0; a < Bc * (j+1) && b < Bc;a++,b++){
                    for(int k = 0; k < d;k++){
                        Kj[(b) * d + k] = K[b * H * N * d + h * N * d + a * d + k];
                        Vj[(b) * d + k] = V[b * H * N * d + h * N * d + a * d + k];
                    }
                }
                for(int i = 0; i < Tr;i++){
                    for(int a = Br * i,b = 0;a < Br *(i+1) && b < Br;a++,b++ ){
                        for(int k = 0; k < d;k++){
                            Qi[b*d+k] = Q[b * H * N * d + h * N * d+a*d + k];
                            Oi[b*d+k] = O[b * H * N * d + h * N * d+a*d + k];
                        }
                        li[b] = l[a];
                    }
                    for(int a = 0;a < Br ;a++){
                        float sum = 0.0;
                        for(int b = 0;b < Bc;b++){
                            float tmp = 0.0;
                            for(int k = 0;k < d;k++){
                                tmp += Qi[a * d + k] * Kj[b * d + k];
                            }
                            Sij[a * Bc + b] = tmp;
                            Pij[a * Bc + b] = std::exp(Sij[a * Bc + b]);
                            sum += Pij[a * Bc + b];
                        }
                        lij[a] = sum;
                        lnew[a] = li[a] + lij[a];
                    }
                    
                    for(int a = 0;a < Br;a++){
                        for(int b = 0; b < d;b++){
                            float tmp = 0.0;
                            for(int k = 0; k < Bc;k++){
                                tmp += Pij[a * Bc + k] * Vj[k * d + b];
                            }
                            Oi[a * d + b] = Oi[a * d + b] * li[a];
                            Oi[a * d + b] = Oi[a * d + b] + tmp;
                            Oi[a * d + b] = Oi[a * d + b] / lnew[a];
                        }
                    }
                    for(int a = Br * i,b = 0;a < Br *(i+1) && b < Br;a++,b++ ){
                        for(int k = 0; k < d;k++){
                            O[b * H * N * d + h * N * d+a*d + k] = Oi[b*d+k]; 
                        }
                        l[a] = lnew[b];
                    }
                }
            }
        }
    }
    // -------- YOUR CODE HERE  -------- //


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}
*/
torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::cout<<"b:"<<B<<"h:"<<H<<"N:"<<N<<"d:"<<d<<std::endl;
    // -------- YOUR CODE HERE  -------- //
// #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i0 = 0; i0 < N; i0+=Br) {
                int _Br = std::min(Br, N - i0);
                // load a tile from Q to Qi
                auto Qtile = QTensor.index({b, h, torch::indexing::Slice(i0, i0 + _Br)});
                std::vector<float> Qi = formatTensor(Qtile);        // (Br, d)

                // allocate buffers for Oi, li
                std::vector<float> Oi = formatTensor(OiTensor);     // (Br, d)
                std::vector<float> li = formatTensor(LiTensor);     // (Br)

                // ;    // (Bc, d)
                // ;    // (Bc, d)

                for (int i1 = 0; i1 < N; i1+=Bc) {
                    int _Bc = std::min(Bc, N - i1);
                    // load a tile from K and V
                    auto Ktile = KTensor.index({b, h, torch::indexing::Slice(i1, i1 + _Bc)});
                    auto Vtile = VTensor.index({b, h, torch::indexing::Slice(i1, i1 + _Bc)});
                    std::vector<float> Kj = formatTensor(Ktile);    // (Bc, d)
                    std::vector<float> Vj = formatTensor(Vtile);    // (Bc, d)

                    // allocate buffers for Sij, Pij, lij, lnew
                    std::vector<float> Sij = formatTensor(SijTensor);   // (Br, Bc)
                    std::vector<float> Pij = formatTensor(PijTensor);   // (Br, Bc)
                    std::vector<float> lij = formatTensor(LijTensor);   // (Br)
                    std::vector<float> lnew = formatTensor(LnewTensor); // (Br)
                    
                    // Sij = QiKj_T
                    // MM2d_v(Qi.data(), /*transA=*/false, 
                    //        Kj.data(), /*transB=*/true, 
                    //        Sij.data(), _Br, _Bc, d);
                    // MM2d(Qi, /*transA=*/false, 
                    //        Kj, /*transB=*/true, 
                    //        Sij, _Br, _Bc, d);
                    
                    trans01_matrix_mul_cuda(Qi.data(),Kj.data(),Sij.data(), _Br, _Bc, d);
                    // general_mul_cuda(Qi.data(),false,Kj.data(),true,Sij.data(), _Br, _Bc, d);

                    // Pij = exp(Sij)
                    for (int ir = 0; ir < _Br; ir++)
                        for (int ic = 0; ic < _Bc; ic++) {
                            float s = exp(twoDimRead(Sij, ir, ic, _Bc));
                            twoDimWrite(Pij, ir, ic, _Bc, s);
                        }      
                
                    // lij = rowSum(Pij)
                    for (int ir = 0; ir < _Br; ir++) {
                        std::vector<float> row(Pij.begin() + ir * _Bc, Pij.begin() + ir * _Bc + _Bc);
                        // lij[ir] = rowSum_v(row.data(), _Bc);
                        float tmp = 0;
                        for(auto r : row){
                            tmp += r;
                        }
                        lij[ir] = tmp;
                    }
                        

                    // lnew = li + lij
                    for (int ir = 0; ir < _Br; ir++)
                        lnew[ir] = li[ir] + lij[ir];
                    
                    // Oi = (liOi + PijVj) / lnew
                    for (int ir = 0; ir < _Br; ir++)
                        for (int j = 0; j < d; j++) {
                            float oi = li[ir] * twoDimRead(Oi, ir, j, d);
                            twoDimWrite(Oi, ir, j, d, oi);
                        }
                    
                    // MM2d_v(Pij.data(), /*transA=*/false, 
                    //        Vj.data(), /*transB=*/false, 
                    //        Oi.data(), _Br, d, _Bc);

                    // MM2d(Pij, /*transA=*/false, 
                    //     Vj, /*transB=*/false, 
                    //     Oi, _Br, d, _Bc);
                    // std::cout<<"cuda X: ";

                    // for(int i = 0; i < 10; i++) {
                    //     std::cout << Pij[i] << " ";
                    // }
                    // std::cout<<""<<std::endl;

                    matrix_mul_cuda(Pij.data(),Vj.data(),Oi.data(),_Br, d, _Bc);
                    
                    for (int ir = 0; ir < _Br; ir++)
                        for (int j = 0; j < d; j++) {
                            float oi = twoDimRead(Oi, ir, j, d) / lnew[ir];
                            twoDimWrite(Oi, ir, j, d, oi);
                        }

                    li = lnew;  // update li
                }

                // write Oi to memory
                int offset = b * H * N * d + h * N * d + i0 * d;
                std::copy(Oi.begin(), Oi.begin() + _Br*d, O.begin() + offset);
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}

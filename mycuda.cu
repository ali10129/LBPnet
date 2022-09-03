#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>




__global__ void LBP_forward_cuda_kernel(
const torch::PackedTensorAccessor32<float,4> input_padded,
const torch::PackedTensorAccessor32<int32_t,3> kernel,
const torch::PackedTensorAccessor32<int32_t,2> map,
torch::PackedTensorAccessor32<float,4> output,
torch::PackedTensorAccessor32<float,5> output0,
 uint32_t n_i,
 uint32_t c_i,  
 uint32_t h_i, 
 uint32_t w_i, 
 uint32_t n_filter,
 uint32_t n_points, 
 uint32_t padwh
) 
{
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t h = blockIdx.y * blockDim.y + threadIdx.y;

    const uint32_t b = blockIdx.z * blockDim.z + threadIdx.z;

    ////////////

    // __shared__ uint32_t kernels[n_filter0][n_points0][2];
    // __shared__ uint32_t maps[n_filter0][n_points0];

    // if(h < n_filter && w < n_points){
    //     kernels[h][w][0] = kernel[h][w][0];
    //     kernels[h][w][1] = kernel[h][w][1];
    //     maps[h][w] = map[h][w];
    // }

    // __syncthreads();

    for (uint32_t q = 0; q < n_filter;q++){
        
        #pragma unroll
        for(uint32_t p = 0; p < n_points;p++){
            float pivot = input_padded[b][map[q][p]][h + padwh][w + padwh];
            float comp = input_padded[b][map[q][p]][h + kernel[q][p][1]][w + kernel[q][p][0]];
            uint32_t tmp = 0;
            if (comp  > pivot) {
                tmp = 1 << p;
            }
            output[b][q][h][w] += tmp;
            output0[b][q][h][w][p] = comp - pivot;
        }
        __syncthreads();
    }

}

std::vector<torch::Tensor>  LBP_forward_cuda(
                    torch::Tensor input,
                    uint32_t padwh,
                    torch::Tensor kernel,
                    torch::Tensor map,
                    torch::Tensor output,
                    torch::Tensor output0,
                    torch::Tensor Image_grad_Y,
                    torch::Tensor Image_grad_X
                    )
    {

        const uint32_t n_i =input.size(0);
        const uint32_t c_i = input.size(1); 
        const uint32_t h_i = input.size(2);
        const uint32_t w_i = input.size(3);
        const uint32_t n_filter = kernel.size(0);
        const uint32_t n_points = kernel.size(1);

        torch::Tensor input_padded = torch::constant_pad_nd(input, {padwh, padwh, padwh, padwh}, 0);

        const uint32_t thx = w_i <= 32 ? w_i : 32;
        const uint32_t thy = h_i <= 32 ? h_i : 32;

        const dim3 threads(thx, thy, 1);
        const dim3 blocks((w_i - 1) / thx + 1, (h_i - 1) / thy + 1, n_i);
        //,n_filter*n_points*3*sizeof(float)
        LBP_forward_cuda_kernel<<<blocks, threads>>>(             
                input_padded.packed_accessor32<float,4>(),
                kernel.packed_accessor32<int32_t,3>(),
                map.packed_accessor32<int32_t,2>(),
                output.packed_accessor32<float,4>(),
                output0.packed_accessor32<float,5>(),
                n_i,
                c_i,  
                h_i, 
                w_i, 
                n_filter,
                n_points, 
                padwh);

        return {output,input_padded};
    }



__device__ __forceinline__ float TanhI(float Out0, uint32_t index_p, const float alpha) {
    float t = tanh(Out0 / alpha);
    t *= t;
    return ((1 << index_p)/ alpha) * (1-t);
}


__global__ void LBP_backward_cuda_kernel(
const torch::PackedTensorAccessor32<float,4> grad_out,
const torch::PackedTensorAccessor32<float,5> output0,
const torch::PackedTensorAccessor32<float,4> image_grad_X_padded,
const torch::PackedTensorAccessor32<float,4> image_grad_Y_padded,
const torch::PackedTensorAccessor32<int32_t,3> kernel,
const torch::PackedTensorAccessor32<int32_t,2> map,
torch::PackedTensorAccessor32<float,4> grad_in_padded,
torch::PackedTensorAccessor32<float,3> grad_kernels,
const uint32_t n_filter,
const uint32_t n_points, 
const float ALPHA
) 
{
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t h = blockIdx.y * blockDim.y + threadIdx.y;

    const uint32_t b = blockIdx.z * blockDim.z + threadIdx.z;

    ////////////

    // __shared__ uint32_t kernels[n_filter][n_points][2];
    // __shared__ uint32_t maps[n_filter][n_points];

    // if(h < n_filter && w < n_points){
    //     kernels[h][w][0] = kernel[h][w][0];
    //     kernels[h][w][1] = kernel[h][w][1];
    //     maps[h][w] = map[h][w];
    // }

    // __syncthreads();

    for (uint32_t q = 0; q < n_filter;q++){
        
        #pragma unroll
        for(uint32_t p = 0; p < n_points;p++){
            float tmp = grad_out[b][q][h][w] * TanhI(output0[b][q][h][w][p], p, ALPHA);

            // grad_in[b][maps[q][p]][h + kernels[q][p][1]][w + kernels[q][p][0]] += tmp;
            atomicAdd(&grad_in_padded[b][map[q][p]][h + kernel[q][p][1]][w + kernel[q][p][0]], tmp);

            // grad_kernels[q][p][0] += tmp * image_grad_X_padded[b][maps[q][p]][h + kernels[q][p][1]][w + kernels[q][p][0]];
            atomicAdd(&grad_kernels[q][p][0], tmp * image_grad_X_padded[b][map[q][p]][h + kernel[q][p][1]][w + kernel[q][p][0]]);

            // grad_kernels[q][p][1] += tmp * image_grad_Y_padded[b][maps[q][p]][h + kernels[q][p][1]][w + kernels[q][p][0]];
            atomicAdd(&grad_kernels[q][p][1], tmp * image_grad_Y_padded[b][map[q][p]][h + kernel[q][p][1]][w + kernel[q][p][0]]);
        }
        __syncthreads();
    }
}


std::vector<torch::Tensor>  LBP_backward_cuda(
        const torch::Tensor grad_out,
        const torch::Tensor output0,
        const torch::Tensor image_grad_X_padded,
        const torch::Tensor image_grad_Y_padded,
        const torch::Tensor kernel,
        const torch::Tensor map,
        torch::Tensor grad_in_padded,
        torch::Tensor grad_kernels,
        const uint32_t n_i,
        const uint32_t c_i,  
        const uint32_t h_i, 
        const uint32_t w_i, 
        const uint32_t n_filter,
        const uint32_t n_points, 
        const float ALPHA)
    {


        const int thx = w_i <= 32 ? w_i : 32;
        const int thy = h_i <= 32 ? h_i : 32;

        const dim3 threads(thx, thy, 1);
        const dim3 blocks((w_i - 1) / thx + 1, (h_i - 1) / thy + 1, n_i);

        LBP_backward_cuda_kernel<<<blocks, threads>>>(
            grad_out.packed_accessor32<float, 4>(),
            output0.packed_accessor32<float, 5>(),
            image_grad_X_padded.packed_accessor32<float, 4>(),
            image_grad_Y_padded.packed_accessor32<float, 4>(),
            kernel.packed_accessor32<int32_t, 3>(),
            map.packed_accessor32<int32_t, 2>(),
            grad_in_padded.packed_accessor32<float, 4>(),
            grad_kernels.packed_accessor32<float, 3>(),
            n_filter,
            n_points,
            ALPHA);

        return {grad_in_padded,grad_kernels};
    }
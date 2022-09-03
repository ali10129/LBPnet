#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> LBP_forward_cuda(
    torch::Tensor input,
    uint32_t padwh,
    torch::Tensor kernel,
    torch::Tensor map,
    torch::Tensor output,
    torch::Tensor output0,
    torch::Tensor Image_grad_Y,
    torch::Tensor Image_grad_X);

std::vector<torch::Tensor> LBP_backward_cuda(
    const torch::Tensor grad_out,
    const torch::Tensor output0,
    const torch::Tensor image_grad_X_padded,
    const torch::Tensor image_grad_Y_padded,
    const torch::Tensor kernel,
    const torch::Tensor map,
    torch::Tensor grad_in_padded,
    torch::Tensor grad_kernels,
    uint32_t n_i,
    uint32_t c_i,
    uint32_t h_i,
    uint32_t w_i,
    uint32_t n_filter,
    uint32_t n_points,
    float ALPHA);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> LBP_forward(
    torch::Tensor input,
    uint32_t padwh,
    torch::Tensor kernel,
    torch::Tensor map,
    torch::Tensor output,
    torch::Tensor output0,
    torch::Tensor Image_grad_Y,
    torch::Tensor Image_grad_X){

        CHECK_INPUT(input);

        CHECK_INPUT(kernel);
        CHECK_INPUT(map);
        CHECK_INPUT(input);
        CHECK_INPUT(output);
        CHECK_INPUT(output0);
        CHECK_INPUT(Image_grad_Y);
        CHECK_INPUT(Image_grad_X);

        return LBP_forward_cuda(
            input,
            padwh,
            kernel,
            map,
            output,
            output0,
            Image_grad_Y,
            Image_grad_X);
    }

std::vector<torch::Tensor> LBP_backward(
    const torch::Tensor grad_out,
    const torch::Tensor output0,
    const torch::Tensor image_grad_X_padded,
    const torch::Tensor image_grad_Y_padded,
    const torch::Tensor kernel,
    const torch::Tensor map,
    torch::Tensor grad_in_padded,
    torch::Tensor grad_kernels,
    uint32_t n_i,
    uint32_t c_i,
    uint32_t h_i,
    uint32_t w_i,
    uint32_t n_filter,
    uint32_t n_points,
    float ALPHA){

        CHECK_INPUT(grad_out);
        CHECK_INPUT(output0);
        CHECK_INPUT(image_grad_X_padded);
        CHECK_INPUT(image_grad_Y_padded);
        CHECK_INPUT(kernel);
        CHECK_INPUT(map);
        CHECK_INPUT(grad_in_padded);
        CHECK_INPUT(grad_kernels);
        return LBP_backward_cuda(
        grad_out,
        output0,
        image_grad_X_padded,
        image_grad_Y_padded,
        kernel,
        map,
        grad_in_padded,
        grad_kernels,
        n_i,
        c_i,
        h_i,
        w_i,
        n_filter,
        n_points,
        ALPHA);
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &LBP_forward, "MY LBP forward (CUDA)");
  m.def("backward", &LBP_backward, "MY LBP backward (CUDA)");
}

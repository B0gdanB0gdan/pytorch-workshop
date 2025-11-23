#include <torch/extension.h>
#include <vector>


__global__ void linear_forward_kernel(float *X, float *W, float *b, float *Y, int in_features, int out_features){

    if (threadIdx.x < out_features){
        float sum = 0;   
        for (int i = 0; i < in_features; i++){
            sum+= X[blockIdx.x * in_features + i] * W[threadIdx.x * in_features + i]; 
        }
        Y[blockIdx.x * out_features + threadIdx.x] = sum + b[threadIdx.x];
    }
}


at::Tensor linear_forward_cuda(at::Tensor X, at::Tensor W, at::Tensor b){

    int batch_size = X.size(0);

    // W: out_features x in_features for simplicity
    int in_features = W.size(1);
    int out_features = W.size(0);

    // distribute threads over neurons
    // distribute blocks over samples

    at::Tensor Y = at::zeros({batch_size, out_features}, X.options());

    linear_forward_kernel<<<batch_size, out_features>>>(X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), Y.data_ptr<float>(), in_features, out_features);
    return Y;
}


__global__ void grad_W_kernel(float *dY, float *X, float *dW, int in_features, int out_features, int batch_size){
    float sum = 0;
    for (int i = 0; i < batch_size; i++){
        sum += dY[i * out_features + blockIdx.x] * X[i * in_features + threadIdx.x];
    }
    dW[blockIdx.x * in_features + threadIdx.x] = sum;
}


__global__ void grad_b_kernel(float *dY, float *db, int out_features, int batch_size){
    float sum = 0;
    for (int i = 0; i < batch_size; i++){
        sum += dY[blockIdx.x + out_features * i];
    }
    db[blockIdx.x] = sum;
}


__global__ void grad_X_kernel(float *dY, float *W, float *dX, int in_features, int out_features, int batch_size){
    float sum = 0;
    for (int i = 0; i < out_features; i++){
        sum += dY[blockIdx.x * out_features + i] * W[i * in_features + threadIdx.x];
    }
    dX[blockIdx.x * in_features + threadIdx.x] = sum;
}


std::vector<at::Tensor> linear_backward_cuda(at::Tensor dY, at::Tensor X, at::Tensor W, at::Tensor b){

    int batch_size = X.size(0);
    int in_features = W.size(1);
    int out_features = W.size(0);

    at::Tensor dW = at::zeros_like(W);
    at::Tensor db = at::zeros_like(b);
    at::Tensor dX = at::zeros_like(X);

    grad_W_kernel<<<out_features, in_features>>>(dY.data_ptr<float>(), X.data_ptr<float>(), dW.data_ptr<float>(), in_features, out_features, batch_size);
    grad_b_kernel<<<out_features, 1>>>(dY.data_ptr<float>(), db.data_ptr<float>(), out_features, batch_size);
    grad_X_kernel<<<batch_size, in_features>>>(dY.data_ptr<float>(), W.data_ptr<float>(), dX.data_ptr<float>(), in_features, out_features, batch_size);

    return {dX, dW, db};
    
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward_cuda, "Linear forward (CUDA)");
  m.def("backward", &linear_backward_cuda, "Linear backward (CUDA)");
}
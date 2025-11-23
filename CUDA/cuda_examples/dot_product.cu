#include <stdio.h>


#define N 1024 * 5
#define threadsPerBlock 1024
#define numBlocks ceil(N / threadsPerBlock)


__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock]; // shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = temp;
    __syncthreads(); // synchronize threads in this block

    int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0){
        c[blockIdx.x] = cache[0];
    }
}

int main() {
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(numBlocks * sizeof(float));

    for(int i=0; i<N; i++){
        a[i] = 2*i;
        b[i] = 2*i+1;
    }

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, numBlocks * sizeof(float));

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    int c = 0;
    for(int i=0; i<numBlocks; i++){
        c += partial_c[i];
    }

    printf("Dot product: %d\n", c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
    
    return 0;
}
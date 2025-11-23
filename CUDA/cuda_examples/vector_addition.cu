#include <stdio.h>
#include <chrono>

#define N 1e8
#define threadsPerBlock 1024
#define numBlocks ceil(N / threadsPerBlock)


void vec_add_cpu(){
    float *a, *b, *c;
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    clock_t start, end;
    start = clock();

    for(int i=0; i<N; i++){
        a[i] = 2*i;
        b[i] = 2*i+1;
    }
    
    end = clock();  
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("Elapsed time sequential code: %f seconds\n", time_taken);

    for(int i=0; i<N; i++){
        c[i] = a[i] + b[i];
    }

    free(a);
    free(b);
    free(c);
}


__global__ void add(float *a, float *b, float *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}


void vec_add_gpu(){
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    for(int i=0; i<N; i++){
        a[i] = 2*i;
        b[i] = 2*i+1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_c, N * sizeof(float));

    cudaEventRecord(start, 0);

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    add<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time parallel code: %f seconds\n", elapsedTime / 1000);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
}

int main() {

    vec_add_cpu();
    vec_add_gpu();

    
    return 0;
}
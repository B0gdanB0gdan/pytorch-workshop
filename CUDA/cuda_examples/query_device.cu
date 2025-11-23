#include <stdio.h>


int main(){

    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for(int i=0; i<count; i++){
        cudaGetDeviceProperties(&prop, i);
        printf("--General Information for device %d--\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor); // major: architecture family e.g., Ampere, minor: incremental improvements
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if(prop.deviceOverlap){
            printf("Enabled\n");
        } else {
            printf("Disabled\n");
        }
        printf("\n");
        printf("--Memory information for device %d--\n", i);
        printf("Total global mem: %zd\n", prop.totalGlobalMem);
        printf("Total constant Mem: %zd\n", prop.totalConstMem);
        printf("\n");
        printf("--MP information for device %d--\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %zd\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max blocks per mp: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n\n");

        /*
            The number of threads per block can be derived from the warpSize, the
            maxThreadsPerBlock, and the register and shared memory demands per thread
            of the kernel, given the sharedMemPerBlock and regsPerBlock fields of the
            cudaDeviceProp structure
        */ 
        int N = 1000000;
        int threads_per_block = prop.warpSize * ceil(prop.maxThreadsPerBlock / prop.warpSize);
        int blocks = ceil(N / threads_per_block); // 1000000 / 1024 = 976.56 ~ 977 > 20 (ideally multiple of SM but "OK")
        printf("E.g. N=1_000_000 => Ideal grid size <<<blocks, threads>>> = %d, %d\n", threads_per_block, blocks);
    }

    

    return 0;
}
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA Kernel for GELU
__global__ void gelu_kernel(float *d_input, float *d_output, int size) {
    int = tod = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float x = d_input[tid];
        d_output[tid] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3))));        
    }
}


// Host side GELU Function
void gelu(float *h_input, float *h_output, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));


    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);


    // Copy Data from device to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    const int size = 10;
    float h_input[size], h_output[size];

    // Initialize input dat a
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i - 5.0f; // Sample data [-5, -4, -3, ..., 4]
    }

    // Apply GELU
    gelu(h_input, h_output, size);

    // Print the results
    for (int i = 0; i < size; i++) {
        printf("Input: %f, GELU: %f\n", h_input[i], h_output[i]);
    }

    return 0;
}
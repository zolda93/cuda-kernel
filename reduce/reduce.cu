#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int recursiveReduce(int *data,int const size)
{
	if (size==1) return data[0];
	int const stride = size/2;
	for(int i =0;i<stride;i++)
	{
		data[i] += data[i+stride];
	}
	return recursiveReduce(data,stride);
}

//Neighbored pair implementation with divergence
__global__ void reduceNeighbored(int *g_idata,int *g_odata,unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int *idata = g_idata + blockIdx.x * blockDim.x;

	if(idx >= n) return;

	for(int stride = 1;stride < blockDim.x; stride *= 2)
	{
		if(tid % (2 * stride) == 0)
		{
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}
	if(tid ==0)g_odata[blockIdx.x] = idata[0];
}

int main(int argc,char **argv)
{
	bool result = false;
	double iStart,iElaps;
	int gpu_sum = 0;
	int size = 1<<14;
	printf("array size %d \n",size);
	int blocksize = 512;
	if(argc > 1) blocksize = atoi(argv[1]);
	dim3 block(blocksize,1);
	dim3 grid((size + block.x - 1) / block.x,1);

	//allocate host memory
	size_t nBytes = size * sizeof(int);
	int *h_idata = (int *)malloc(nBytes);
	int *h_odata = (int *)malloc(grid.x * sizeof(int));
	int *tmp = (int *)malloc(nBytes);

	//initialize the array
	for(int i=0;i<size;i++)
	{
		h_idata[i] = (int)( rand() & 0xFF );
	}
	memcpy(tmp,h_idata,nBytes);
	
	//allocate device memory

	int *d_idata = NULL;
	int *d_odata = NULL;
	cudaMalloc((void **)&d_idata,nBytes);
	cudaMalloc((void **)&d_odata,grid.x * sizeof(int));

	// cpu reduction
    	iStart = seconds();
    	int cpu_sum = recursiveReduce (tmp, size);
    	iElaps = seconds() - iStart;
    	printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    	// kernel 1: reduceNeighbored
    	cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    	cudaDeviceSynchronize();
    	iStart = seconds();
    	reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    	cudaDeviceSynchronize();
    	iElaps = seconds() - iStart;
    	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    	gpu_sum = 0;
    	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    	printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block ""%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

	// free host memory
    	free(h_idata);
    	free(h_odata);
    	free(tmp);

    	// free device memory
    	cudaFree(d_idata);
    	cudaFree(d_odata);

    	// reset device
    	cudaDeviceReset();

    	// check the results
    	result = (gpu_sum == cpu_sum);

    	if(!result) printf("Test failed!\n");

    	return EXIT_SUCCESS;
}




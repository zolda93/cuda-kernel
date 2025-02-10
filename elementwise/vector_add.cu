#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void add(float *a,float *b,float *c)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[idx] = a[idx] + b[idx];
}

__global__ void add2(float *a,float *b,float *c)
{
	unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
	float2 reg_a = FETCH_FLOAT2(a[idx]);
	float2 reg_b = FETCH_FLOAT2(b[idx]);
	float2 reg_c;
	reg_c.x = reg_a.x + reg_b.x;
	reg_c.y = reg_a.y + reg_b.y;
	FETCH_FLOAT2(c[idx]) = reg_c;
}

__global__ void add4(float *a,float *b,float *c)
{
	unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	float4 reg_a = FETCH_FLOAT4(a[idx]);
	float4 reg_b = FETCH_FLOAT4(b[idx]);
	float4 reg_c;
	reg_c.x = reg_a.x + reg_b.x;
	reg_c.y = reg_a.y + reg_b.y;
	reg_c.z = reg_a.z + reg_b.z;
	reg_c.w = reg_a.w + reg_b.w;
	FETCH_FLOAT4(c[idx]) = reg_c;
}

void check(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}


void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void add_host(float *a, float *b, float *c, const int N)
{
    for (int idx = 0; idx < N; idx++)
        c[idx] = a[idx] + b[idx];
}

int main(int argc,char **argv)
{
	const int N = 32*1024*1024;
	double iElaps,iStart;
	float *a = (float *)malloc(N*sizeof(float));
	float *b = (float *)malloc(N*sizeof(float));
	float *c = (float *)malloc(N*sizeof(float));
	float *res = (float *)malloc(N*sizeof(float));
	float *d_a,*d_b,*d_c;
	cudaMalloc((float**)&d_a,N*sizeof(float));
	cudaMalloc((float**)&d_b,N*sizeof(float));
	cudaMalloc((float**)&d_c,N*sizeof(float));

	initialData(a,N);
	initialData(b,N);

	add_host(a,b,c,N);

	cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);

	//dim3 block(THREAD_PER_BLOCK,1);
	//dim3 grid(N/THREAD_PER_BLOCK,1);
	//iStart = seconds();
	//add<<<grid,block>>>(d_a,d_b,d_c);
	//iElaps = seconds() - iStart;
	//printf("naiv add  elapsed %f sec\n", iElaps);

	//dim3 block(THREAD_PER_BLOCK,1);
        //dim3 grid(N/THREAD_PER_BLOCK/2,1);
        //iStart = seconds();
        //add2<<<grid,block>>>(d_a,d_b,d_c);
        //iElaps = seconds() - iStart;
        //printf("add2  elapsed %f sec\n", iElaps);

	dim3 block(THREAD_PER_BLOCK,1);
	dim3 grid(N/THREAD_PER_BLOCK/4,1);
	iStart = seconds();
	add4<<<grid,block>>>(d_a,d_b,d_c);
	iElaps = seconds() - iStart;
        printf("add4  elapsed %f sec\n", iElaps);

	cudaMemcpy(res,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);
	check(res,c,N);
	free(a);
	free(b);
	free(c);
	free(res);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}




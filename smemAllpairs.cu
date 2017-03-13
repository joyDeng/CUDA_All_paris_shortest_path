#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>


//C = A * B;
//kernel function to calculate each position of C
float * Sequential(float *L, int n);
float * shortestPaths(float *A, float *B,int n);
template <int BLOCK_SIZE> 
__global__
void Mamulti(int n , float *A, float *B, float *C){
	//block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//thread.index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//begin of vector a;
	int Abegin = by * BLOCK_SIZE * n;

	//step of submatrix each time the block move right forward
	int Astep = BLOCK_SIZE;

	//last submatrix of a
	int Aend = Abegin + n - 1;

	//begin of b
	int Bbegin = bx * BLOCK_SIZE;

	//step of 'b' each time the block move down
	int Bstep = BLOCK_SIZE * n;
	int index = BLOCK_SIZE * by * n + BLOCK_SIZE * bx;

	float tempc = FLT_MAX;

	for(int a = Abegin,b = Bbegin;
		a <= Aend;
		a += Astep,b += Bstep){

		//Declare shared mem for array a
		__shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];

		//Declare shared mem for array b
		__shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

		//load data from device memory to shared memory
		//each thread responsible for one position in block
		Asub[ty][tx] = A[a + n * ty + tx];
		Bsub[ty][tx] = B[b + n * ty + tx];

		//Make sure the matrices are loaded
        __syncthreads();

        // Multiply the submatrixes and thus gain submatrix of c
        for(int k = 0; k < BLOCK_SIZE ; k++)
        	tempc = min(tempc,Asub[ty][k] + Bsub[k][tx]);

        __syncthreads();
        // Make sure each position has finished multiplication
	}

	//Write element to result matrix C.
	//each thread writes one position.
	C[index + ty * n + tx] = tempc;
}

//generate graph randomly
float* generateGraph(int n, int range, float p){
	float* l = (float *)malloc(n*n*sizeof(float));
	
	//set random seed
	int seed = time(NULL);
	srand(seed);

	//generate random seed
	for(int j = 0 ; j < n ; ++j){
		for(int i = 0 ; i < n ; ++i){
			float po = ((float)rand()/(float)(RAND_MAX));
			//printf("%f\n", po);
			 if(po > p){
			  l[j*n+i] = 1;//(float)(rand()%range)+1;
			 }
			 else l[j*n+i] = FLT_MAX;

			//printf("%f\n", FLT_MAX );
		}
		l[j*n+j] = 0;//form vertex to itself is 0
	}
	return l;
}

int main(void){
	int n = 1<<14;
	int N = 1<<28;
	int m = 1;
	int range = 100;
	//int wrongAnswer = 0;
	float time_cuda = 0;
	float *L,*B, *d_a, *d_b, *S;
	//cudaEvent_t start,stop;
	struct timeval t1,t2;
	struct timeval tp1,tp2;
	
    printf("Problam Matrix size: %d X %d\n",n,n );
	//allocate memory on the host
	//initialize dataon the host
	//genreate graph randomly
	//L = generateGraph(n,range,0.5);

	L = (float *)malloc(N*sizeof(float));
	S = (float *)malloc(N*sizeof(float));
	//B = (float *)malloc(N*sizeof(float));
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++){
			if(i == j-1) L[i*n+j] = 1;
			else L[i*n+j] = FLT_MAX;
		}
		L[i*n+i] = 0; 
	}
	for(int i = 0 ; i < N ; i++){
		//B[i] = FLT_MAX;	
		S[i] = L[i];
		//printf("%f\n", L[i]);
	}

	// for(int i = 0 ; i < n ; i++)
	// printf("%f\n",L[i]);
    
	//printf("%f\n", min(FLT_MAX, FLT_MAX + FLT_MAX) );
	//allocate unified memory accessible for GPU and CPU
	cudaMalloc(&d_a,N*sizeof(float));//source
	cudaMalloc(&d_b,N*sizeof(float));//result
	//cudaMalloc(&C,N*sizeof(float));

	//copy data from host to devices
	cudaMemcpy(d_a,L,N*sizeof(float),cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice);

	//do matrixmulti
	int blockSize = 16;

	dim3 threads(blockSize, blockSize);
    dim3 grid(n / threads.x, n / threads.y);

    while(m < n-1){
       gettimeofday(&tp1,0);
	   Mamulti<16><<<grid,threads>>>(n,d_a,d_a,d_b);
	   //cudaDeviceSynchronize();
	   gettimeofday(&tp2,0);
	   cudaMemcpy(d_a,d_b,N*sizeof(float),cudaMemcpyDeviceToDevice);
	   //cudaMemcpy(B,d_a,N*sizeof(float),cudaMemcpyHostToDevice);
	   time_cuda += (1000000.0*(tp2.tv_sec - tp1.tv_sec) + tp2.tv_usec-tp1.tv_usec)/1000;
       m = 2*m;
    }
    cudaMemcpy(L,d_b,N*sizeof(float),cudaMemcpyDeviceToHost);

    //compute sequential result
	//timing
	
    
    printf("================================================\n");
	printf("Parallel Shared_Mem use :%fms\n", time_cuda);

	// gettimeofday(&t1,0);

 //    S = Sequential(S,n);
    

	// gettimeofday(&t2,0);

	// for(int i = 0 ; i < N ; i++) 
	//   if(abs(S[i] - B[i]) > 0.00001) {
	//   	printf("position i wrong\n");
	//   	wrongAnswer = 1;
	//   }
	// float time = (1000000.0*(t2.tv_sec - t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000;
    
	// printf("Sequential use :%fms\n", time);
	// printf("================================================\n");
	// if(wrongAnswer) printf("Bad Answer\n");
	// else printf("Answer OK\n");
    //free allocated memory on both host and device
	cudaFree(d_a);
	cudaFree(d_b);
	free(S);
	free(L);
	//free(B);
	
}

float * Sequential(float *L, int n){
	int m = 1;
	float * L2;
	while(m < n-1){
		//printf("m = %d\n", m);
		L2 = shortestPaths(L,L,n);
		free(L);
		L = L2;
		m = 2*m;
	}
	return L;
}

float * shortestPaths(float *A, float *B,int n){
	float * L2;
	L2  = (float*)malloc(n*n*sizeof(float));
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++ ){
			L2[i*n+j] = FLT_MAX;
		  for(int k = 0 ; k < n ; k++){
			L2[i*n+j] = min(L2[i*n+j],A[i*n+k]+B[k*n+j]);
		  }
		  //printf("L2[%d][%d] = %f\n",i,j,L2[i*n+j]);
		
	    }
    }
    return L2;
}
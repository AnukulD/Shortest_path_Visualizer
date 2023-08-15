#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;
#define TILE_DIM 32
#define thread 32
// kernel for transpose
__global__ void transpose(int r ,int q,int *d_matrixOut, int *d_matrixIn ) {
   
	 int id= blockIdx.y*32*32*gridDim.x +blockIdx.x*32*32 +threadIdx.x +threadIdx.y*32;
     int row= id/q;
	 int col=id%q;
	 int no=q*r;
	 if(id<no)
             d_matrixOut[row+col*r] =d_matrixIn[id];
			
	
}
__global__ void multiply_matrices(int p, int q, int r,int *d_matrixC, int *d_matrixA, int *d_matrixB,int *d_matrixE) {
    __shared__ int A[1024], B[1024];
    int k=0,cValue = 0, x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y,loop=ceil(float(q)/32), Aloc=threadIdx.y+threadIdx.x*32,Bloc=threadIdx.y+threadIdx.x*32;
     while(k<loop) {
		 //checking conditions
         int val1 = k*32 + threadIdx.x;
         if(val1 < q && y < p)
             A[Aloc] = d_matrixA[y*q + val1];
         else
             A[Aloc] = 0;
         if(p < 0)
         {
             for(int kk = 0; kk < cValue; kk++)
             {
                  A[Aloc] = d_matrixA[y*q + k*TILE_DIM + threadIdx.x];
             }
         }
         int val = k*32 + threadIdx.y;
         if(val < q && x < r)
             B[Bloc] = d_matrixB[(val)*r + x];
         else
             B[Bloc] = 0;
         if(q < 0)
         {
             for(int uu = 0; uu < cValue; uu++)
             {
                  B[Bloc] = d_matrixB[(k*32 + threadIdx.y)*r + x];
             }
         }

         __syncthreads();
         int n=0;
         while(n<32)
             {  int a=A[threadIdx.y+n*32] ,b= B[n+threadIdx.x*32];
				cValue += a*b;
                n++;
			 }
         __syncthreads();

		k++; 
    }

    if(y<p && x<r)
        d_matrixE[y*r + x] += cValue;
}



// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE) {
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
	//cudaMemset(dataGPU, 0, 1000*sizeof(int));
	cudaMemset(d_matrixE,0,p*r*sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
   
  //TRANSPOSE OF D --> r*q
  
  int griddiminxdirection =  ceil((float(r))/32);
  int griddiminydirection =  ceil((float(q))/32);
  int griddiminzdirection =  1;
  int blocdiminxdirection =  32;
  int blockdiminydirection=  32;
  int blockdiminzdirection=  1;
  dim3 grid(griddiminxdirection,griddiminydirection,griddiminzdirection);
  dim3 block(blocdiminxdirection,blockdiminydirection,blockdiminzdirection);
  int *Dtranspose;
  int Dtransposesize=r*q*sizeof(int);
  cudaMalloc(&Dtranspose,Dtransposesize);
  transpose<<<grid,block>>>(r,q,Dtranspose,d_matrixD);
  d_matrixD = Dtranspose;
  
    //Matrix mul CDtranspose and AB
  int Griddiminxdirection =  ceil((float(r))/32);
  int Griddiminydirection =  ceil((float(p))/32);
  int Griddiminzdirection =  1;
 

  dim3 Grid(Griddiminxdirection,Griddiminydirection,Griddiminzdirection);
  dim3 Block(blocdiminxdirection,blockdiminydirection,blockdiminzdirection);

  int *CDtranspose;
  int *AB;
  int  matrixsize=p*r*sizeof(int);
 
  cudaMalloc(&CDtranspose,matrixsize);
  multiply_matrices<<<Grid,Block>>>(p,q,r,CDtranspose,d_matrixC,d_matrixD,d_matrixE);
  cudaMalloc(&AB,matrixsize);
  multiply_matrices<<<Grid,Block>>>(p,q,r,AB,d_matrixA,d_matrixB,d_matrixE);
   
   
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	

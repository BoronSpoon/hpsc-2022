#include <cstdio>
#include <cstdlib>

__global__ void initialize(int *bucket, int *sum_bucket) {
  bucket[threadIdx.x] = 0;
  sum_bucket[threadIdx.x] = 0;
}

__global__ void add(int *key, int *bucket) {
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

__global__ void count(int *key, int *bucket, int *sum_bucket) {
  int i = threadIdx.x;
  for (int j = 0; j<i; j++) {
    sum_bucket[i] += bucket[j];
  }
  __syncthreads();
  for (int j = 0; j<bucket[i]; j++){
    key[sum_bucket[i]+j] = i;
  }
  __syncthreads();
}

int main(void) {
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;
  int *sum_bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&sum_bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  
  initialize<<<1,range>>>(bucket,sum_bucket);
  cudaDeviceSynchronize();
  add<<<1,n>>>(key,bucket);
  cudaDeviceSynchronize();
  count<<<1,range>>>(key,bucket,sum_bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(sum_bucket);
}

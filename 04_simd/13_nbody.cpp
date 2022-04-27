#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float reduce_sum(__m256 avec) {
  const int N = 8;
  float a[N];
  for (int i=0; i<N; i++){
    a[i] = 0;
  }
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(a, bvec);
  return a[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], range[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    range[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 rangevec = _mm256_load_ps(range);
  __m256 zerovec = _mm256_setzero_ps();
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec,rangevec,_CMP_NEQ_OQ);
    __m256 rx = _mm256_sub_ps(_mm256_set1_ps(x[i]),xvec);
    __m256 ry = _mm256_sub_ps(_mm256_set1_ps(y[i]),yvec);
    __m256 recpr = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rx,rx),_mm256_mul_ps(ry,ry)));
    __m256 recpr3 = _mm256_mul_ps(recpr,_mm256_mul_ps(recpr,recpr));
    __m256 fxivec = _mm256_blendv_ps(zerovec,_mm256_mul_ps(rx,_mm256_mul_ps(mvec,recpr3)),mask);
    __m256 fyivec = _mm256_blendv_ps(zerovec,_mm256_mul_ps(ry,_mm256_mul_ps(mvec,recpr3)),mask);
    fx[i] -= reduce_sum(fxivec);
    fy[i] -= reduce_sum(fyivec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

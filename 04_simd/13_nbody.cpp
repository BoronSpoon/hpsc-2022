#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __mm256 xvec = _mm256_load_ps(x);
  __mm256 yvec = _mm256_load_ps(y);
  for(int i=0; i<N; i++) {
      if(i != j) {
        __mm256 rx = _mm_sub_ps(_mm_set1_ps(x[i]),x);
        __mm256 ry = _mm_sub_ps(_mm_set1_ps(y[i]),y);
        __mm256 recpr = _mm256_rsqrt_ps(_mm_mul_ps(rx,rx),_mm_mul_ps(ry,ry));
        fx[i] -= rx * m[j] / recpr(r * r * r);
        fy[i] -= ry * m[j] * _mm_mul_ps(_mm_mul_ps(recpr,recpr),recpr);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

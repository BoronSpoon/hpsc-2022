#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <time.h>
using namespace std;
/**************************************** Benchmark on q_core (4 cores) **************************************
nx=ny=41, nt=500, nit=50
- python: 137 s 
    - python 10_cavity_python.py (module: python)
- normal c++: 3.77 s
    - g++ 10_cavity.cpp; ./a.out (module: gcc)
- openmp: 1.61 s
    - g++ 10_cavity_openmp.cpp -fopenmp; ./a.out (module: gcc)
- mpi: 1.08 s (time shown on intel vtune profiler)
    - mpiicpc -O3 10_cavity_mpi.cpp; mpirun -genv VT_LOGFILE_FORMAT=SINGLESTF -trace -n 4 ./a.out 
        - (module: intel intel-mpi intel-itac)
- mpi & openmp: 2.28 s (probably because qnode only has 4 nodes)
    - mpiicpc -O3 -fopenmp 10_cavity_mpi_and_openmp.cpp; mpirun -genv VT_LOGFILE_FORMAT=SINGLESTF -trace -n 4 ./a.out 
        - (module: intel intel-mpi intel-itac)
************************************************************************************************************/
int main(int argc, char** argv) {
    struct timespec tic, toc; // for execution time measurement
    double time = 0; // for execution time measurement
    clock_gettime(CLOCK_REALTIME, &tic);
    MPI_Init(&argc, &argv);
    MPI_Win win0, win1, win2, win3, win4, win5;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nx = 41;
    int ny = 41;
    int nt = 500;
    //int nt = 5; // for printf debug
    int nit = 50;
    int ny_split = 0;
    int count = 0;
    int ny_splits[size]; // ny split amongst ranks ([11,11,11,14] for rank=[0,1,2,3] and ny=41)
    int counts[size]; // number of elements in u,v,p,b that will be gathered to u0,v0,p0,b0 ([9*nx,9*nx,9*nx,14*nx] for rank=[0,1,2,3] and ny=41)
    int displacements[size]; // displacements of u,v,p,v of each rank in u0,v0,p0,b0 ([0,sum(counts[0:1]),sum(counts[0:2]),sum(counts[0:3])] for rank=[0,1,2,3])
    double dx = 2 / (double(nx) - 1);
    double dy = 2 / (double(ny) - 1);
    double dt = 0.01;
    double rho = 1;
    double nu = 0.02;

    // when ny = 41 and size = 4; -> quotient, remainder = 9, 3
    int quotient = double(ny-2)/double(size);
    int remainder = (ny-2) - quotient*size;
    for (int i = 0; i < size; i++) {
        counts[i] = 0;
        ny_splits[i] = 0;
        displacements[i] = 0;
        // rank=0: [0,1,...,10,11                              ], ny_split = 10+2
        // rank=1: [        10,11,...,20,21                    ], ny_split = 10+2
        // rank=2: [                  20,21,...,30,31          ], ny_split = 10+2
        // rank=3: [                            30,31,...,39,40], ny_split =  9+2
        if (i < remainder) { // split the remainder with the first (remainder) ranks
            ny_splits[i] = (quotient+1) + 2; // 2 = before and after elements
        } else {
            ny_splits[i] = (quotient) + 2; // 2 = before and after elements
        }
        // rank=0: [0,1,...,8,9                                            ], counts = (10+2-2)*nx
        // rank=1: [           10,11,...,18,19                             ], counts = (10+2-2)*nx
        // rank=2: [                          20,21,...28,29               ], counts = (10+2-2)*nx
        // rank=3: [                                        30,31,...,39,40], counts = ( 9+2  )*nx
        if (i != size-1) {
            counts[i] = (ny_splits[i] - 2)*nx; // the last two elements are overlapping with adjacent ranks so will be removed for MPI_Gatherv()
        } else {
            counts[i] = (ny_splits[i])*nx;
        }
        // rank=0: counts = 10*nx, displacements = 0
        // rank=1: counts = 10*nx, displacements = 0 + 10*nx
        // rank=2: counts = 10*nx, displacements = 0 + 10*nx + 10*nx
        // rank=3: counts = 11*nx, displacements = 0 + 10*nx + 10*nx + 10*nx
        if (i == 0) { 
            displacements[i] = 0;
        } else {
            displacements[i] = displacements[i-1] + counts[i-1];
        }
    }
    count = counts[rank];
    ny_split = ny_splits[rank]; 
    // np.zeros() default dtype float64 = double (in c) 
    // np.zeros() -> vector defaults to zero in c
    // data split among nodes
    vector<double> u(ny_split*nx);
    vector<double> v(ny_split*nx);
    vector<double> p(ny_split*nx);
    vector<double> b(ny_split*nx);
    // stores data gathered from nodes and processed in rank=0
    vector<double> u0(ny*nx);
    vector<double> v0(ny*nx);
    vector<double> p0(ny*nx);
    vector<double> b0(ny*nx);
    printf("rank: %d, ny_split:%d, count:%d, displacement:%d\n", rank, ny_split, counts[rank], displacements[rank]); // debug
    for (int n = 0; n < nt; n++) {
#pragma omp parallel for collapse(2)
        for (int j = 1; j < ny_split-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                b[j*nx + i] = rho * (
                    1 / dt * ((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx) + (v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy)) -
                    pow(((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx)), 2) - 
                    2 * ((u[(j+1)*nx + i] - u[(j-1)*nx + i]) / (2 * dy) * (v[j*nx + i+1] - v[j*nx + i-1]) / (2 * dx)) - 
                    pow(((v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy)), 2)
                );
            }
        }
        for (int it = 0; it < nit; it++) {
            vector<double> pn = p; // deepcopy
#pragma omp parallel for collapse(2)
            for (int j = 1; j < ny_split-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    p[j*nx + i] = (
                        pow(dy, 2) * (pn[j*nx + i+1] + pn[j*nx + i-1]) +
                        pow(dx, 2) * (pn[(j+1)*nx + i] + pn[(j-1)*nx + i]) -
                        b[j*nx + i] * pow(dx, 2) * pow(dy, 2)
                    ) / (2 * (pow(dx, 2) + pow(dy, 2)));
                }
            }
            
            if (n == 0 && it == 0) {
                MPI_Win_create(&p[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win0);
                MPI_Win_create(&p[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
            }
            MPI_Win_fence(0, win0);
            // rank=0: [0,1,...,10,11                              ], ny_split = 10+2
            // rank=1: [        10,11,...,20,21                    ], ny_split = 10+2
            // ex) for rank 0, send element 10 to rank 1, because element 10 is used but not calculatable in rank 1
            MPI_Put(&p[(ny_split-2)*nx], nx, MPI_DOUBLE, (rank + 1) % size, 0, nx, MPI_DOUBLE, win0); // send p to rank + 1 (including rank = 0 and -1)
            MPI_Win_fence(0, win0);
            MPI_Win_fence(0, win1);
            // ex) for rank 1, send element 11 to rank 0, because element 11 is used but not calculatable in rank 0
            MPI_Put(&p[1*nx], nx, MPI_DOUBLE, (rank - 1 + size) % size, 0, nx, MPI_DOUBLE, win1); // send p to rank - 1 (including rank = 0 and -1)
            MPI_Win_fence(0, win1);
#pragma omp parallel for
            for (int j = 0; j < ny_split; j++) {
                p[j*nx + nx-1] = p[j*nx + nx-2];
                p[j*nx + 0] = p[j*nx + 1];
            }
            if (rank == 0){ // fix the MPI_PUT values for j = 0 (rank = 0)
#pragma omp parallel for
                for (int i = 0; i < nx; i++) p[0*nx + i] = p[1*nx + i];
            } else if (rank == size-1){ // // fix the MPI_PUT values for j = -1 (rank = size-1)
#pragma omp parallel for
                for (int i = 0; i < nx; i++) p[(ny_split-1)*nx + i] = 0;
            }
        }
        // deepcopy
        vector<double> un = u;
        vector<double> vn = v;
#pragma omp parallel for collapse(2)
        for (int j = 1; j < ny_split-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                u[j*nx + i] = un[j*nx + i] 
                    - un[j*nx + i] * dt / dx * (un[j*nx + i] - un[j*nx + i-1])
                    - un[j*nx + i] * dt / dy * (un[j*nx + i] - un[(j-1)*nx + i])
                    - dt / (2 * rho * dx) * (p[j*nx + i+1] - p[j*nx + i-1])
                    + nu * dt / pow(dx, 2) * (un[j*nx + i+1] - 2 * un[j*nx + i] + un[j*nx + i-1])
                    + nu * dt / pow(dy, 2) * (un[(j+1)*nx + i] - 2 * un[j*nx + i] + un[(j-1)*nx + i]);
                v[j*nx + i] = vn[j*nx + i] 
                    - vn[j*nx + i] * dt / dx * (vn[j*nx + i] - vn[j*nx + i-1])
                    - vn[j*nx + i] * dt / dy * (vn[j*nx + i] - vn[(j-1)*nx + i])
                    - dt / (2 * rho * dx) * (p[(j+1)*nx + i] - p[(j-1)*nx + i])
                    + nu * dt / pow(dx, 2) * (vn[j*nx + i+1] - 2 * vn[j*nx + i] + vn[j*nx + i-1])
                    + nu * dt / pow(dy, 2) * (vn[(j+1)*nx + i] - 2 * vn[j*nx + i] + vn[(j-1)*nx + i]);
            }
        }
        if (n == 0) {
            MPI_Win_create(&u[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
            MPI_Win_create(&u[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win3);
            MPI_Win_create(&v[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win4);
            MPI_Win_create(&v[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win5);
        }
        MPI_Win_fence(0, win2);
        MPI_Put(&u[(ny_split-2)*nx], nx, MPI_DOUBLE, (rank + 1) % size, 0, nx, MPI_DOUBLE, win2); // send u to rank + 1 (including rank = 0 and -1)
        MPI_Win_fence(0, win2);
        MPI_Win_fence(0, win3);
        MPI_Put(&u[1*nx], nx, MPI_DOUBLE, (rank - 1 + size) % size, 0, nx, MPI_DOUBLE, win3); // send u to rank - 1 (including rank = 0 and -1)
        MPI_Win_fence(0, win3);
        MPI_Win_fence(0, win4);
        MPI_Put(&v[(ny_split-2)*nx], nx, MPI_DOUBLE, (rank + 1) % size, 0, nx, MPI_DOUBLE, win4);  // send v to rank + 1 (including rank = 0 and -1)
        MPI_Win_fence(0, win4);
        MPI_Win_fence(0, win5);
        MPI_Put(&v[1*nx], nx, MPI_DOUBLE, (rank - 1 + size) % size, 0, nx, MPI_DOUBLE, win5); // send v to rank - 1 (including rank = 0 and -1)
        MPI_Win_fence(0, win5);
#pragma omp parallel for
        for (int j = 0; j < ny_split; j++) {
            u[j*nx + 0]    = 0;
            u[j*nx + nx-1] = 0;
            v[j*nx + 0]    = 0;
            v[j*nx + nx-1] = 0;
        }
        if (rank == 0){
#pragma omp parallel for
            for (int i = 0; i < nx; i++) {
                u[0*nx + i] = 0;
                v[0*nx + i] = 0;
            }
        } else if (rank == size-1){
#pragma omp parallel for
            for (int i = 0; i < nx; i++) {
                u[(ny_split-1)*nx + i] = 1;
                v[(ny_split-1)*nx + i] = 0;
            }
        }
        // gather u,v,p,b results to rank 0. use MPI_Gatherv instead of MPI_Gather because the element count is not equal (10,10,10,11 for size=4,ny=41)
        MPI_Gatherv(&u[0], count, MPI_DOUBLE, &u0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&v[0], count, MPI_DOUBLE, &v0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&p[0], count, MPI_DOUBLE, &p0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&b[0], count, MPI_DOUBLE, &b0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        /*
        if (rank == 0) { // calculate and compare mean & std of u,v,p,b to check that the answer is correct
            double mean_u = 0;
            double mean_v = 0;
            double mean_p = 0;
            double mean_b = 0;

            double mean2_u = 0;
            double mean2_v = 0;
            double mean2_p = 0;
            double mean2_b = 0;

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    mean_u += u0[j*nx + i]/(nx*ny);
                    mean_v += v0[j*nx + i]/(nx*ny);
                    mean_p += p0[j*nx + i]/(nx*ny);
                    mean_b += b0[j*nx + i]/(nx*ny);
                    mean2_u += pow(u0[j*nx + i],2)/(nx*ny);
                    mean2_v += pow(v0[j*nx + i],2)/(nx*ny);
                    mean2_p += pow(p0[j*nx + i],2)/(nx*ny);
                    mean2_b += pow(b0[j*nx + i],2)/(nx*ny);
                }
            }
            double std_u = sqrt(abs(pow(mean_u,2)-mean2_u));
            double std_v = sqrt(abs(pow(mean_v,2)-mean2_v));
            double std_p = sqrt(abs(pow(mean_p,2)-mean2_p));
            double std_b = sqrt(abs(pow(mean_b,2)-mean2_b));

            printf("n: %d\n", n);
            printf("u: mean:%lf, std:%lf\n", mean_u, std_u);
            printf("v: mean:%lf, std:%lf\n", mean_v, std_v);
            printf("p: mean:%lf, std:%lf\n", mean_p, std_p);
            printf("b: mean:%lf, std:%lf\n", mean_b, std_b);
        }*/
    }
    MPI_Win_free(&win0); MPI_Win_free(&win1); MPI_Win_free(&win2); MPI_Win_free(&win3); MPI_Win_free(&win4); MPI_Win_free(&win5);
    MPI_Finalize();
    clock_gettime(CLOCK_REALTIME, &toc);
    time = (toc.tv_sec - tic.tv_sec) + double(toc.tv_nsec - tic.tv_nsec) / double(1000000000L);
    printf("%lf s",time);
} // close int main()

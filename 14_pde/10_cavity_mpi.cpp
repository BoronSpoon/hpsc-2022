#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <mpi.h>
using namespace std;
/************** Benchmark on q_core (4 cores) ****************
nx=ny=41, nt=500, nit=50
- python: 137 s
    - python 10_cavity_python.py
- normal c++: 3.77 s
    - g++ 10_cavity.cpp; ./a.out
- openmp: 1.61 s
    - g++ 10_cavity_openmp.cpp -fopenmp; ./a.out
- mpi: ? s
    - mpicxx 10_cavity_mpi.cpp, mpirun -np 4 ./a.out
*************************************************************/
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
MPI_Win win0;
MPI_Win win1;
MPI_Win win2;
MPI_Win win3;
MPI_Win win4;
MPI_Win win5;
double tic = MPI_Wtime();
int size, rank;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int nx = 41;
int ny = 41;
//int nt = 500;
int nt = 5; // debug
int nit = 50;
int send_to = 0;
int ny_split = 0;
int ny_splits[size]; // length of each split u,v,p,b (the total length >= ny)
int counts[size]; // length of each split u,v,p,b that will be sent to u0,v0,p0,b0 (the total length = ny)
int displacements[size]; // displacements of u,v,p,v of each rank in u0,v0,p0,b0
double dx = 2 / (double(nx) - 1);
double dy = 2 / (double(ny) - 1);
double dt = 0.01;
double rho = 1;
double nu = 0.02;

// split j for MPI
// split j = 1 ~ ny-2 into size

for (int i = 0; i < size; i++) {
    counts[i] = 0;
    ny_splits[i] = 0;
    displacements[i] = 0;
    if (i == size-1) {
        ny_splits[i] = (ny-2) - (size-1)*int(double(ny-2)/double(size)) + 2; // include before and after elements
        counts[i] = ny_splits[i];
    } else {
        ny_splits[i] = double(ny-2)/double(size) + 2; // include before and after elements
        counts[i] = ny_splits[i] - 2; // the last two elements are overlapping with adjacent ranks
    }
    if (i == 0) { 
        displacements[i] = 0;
    } else {
        displacements[i] = displacements[i-1] + counts[i-1];
    }
}
ny_split = ny_splits[rank]; 
// np.zeros() default dtype float64 = double (in c)
// vector defaults to zero
// store split data
vector<double> u(ny_split*nx);
vector<double> v(ny_split*nx);
vector<double> p(ny_split*nx);
vector<double> b(ny_split*nx);
// store all data
vector<double> u0(ny*nx);
vector<double> v0(ny*nx);
vector<double> p0(ny*nx);
vector<double> b0(ny*nx);
printf("rank: %d,ny_split:%d,count:%d,displacement:%d\n", rank, ny_split, counts[rank], displacements[rank]); // debug
for (int n = 0; n < nt; n++) {
    for (int j = 1; j < ny_split-1; j++) {
        for (int i = 1; i < nx-1; i++) { // loop order is already optimal
            b[j*nx + i] = rho * (
                1 / dt * ((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx) + (v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy)) -
                pow(((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx)), 2) - 
                2 * ((u[(j+1)*nx + i] - u[(j-1)*nx + i]) / (2 * dy) * (v[j*nx + i+1] - v[j*nx + i-1]) / (2 * dx)) - 
                pow(((v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy)), 2)
            );
        }
    }
    printf("n:%d,rank:%d,b[0:5]=%lf,%lf,%lf,%lf,%lf\n",n,rank,b[0],b[1],b[2],b[3],b[4])
    for (int it = 0; it < nit; it++) {
        vector<double> pn = p; // deepcopy
        for (int j = 1; j < ny_split-1; j++) {
            for (int i = 1; i < nx-1; i++) { // loop order is already optimal
                p[j*nx + i] = (
                    pow(dy, 2) * (pn[j*nx + i+1] + pn[j*nx + i-1]) +
                    pow(dx, 2) * (pn[(j+1)*nx + i] + pn[(j-1)*nx + i]) -
                    b[j*nx + i] * pow(dx, 2) * pow(dy, 2)
                ) / (2 * (pow(dx, 2) + pow(dy, 2)));
            }
        }
        for (int j = 1; j < ny_split-1; j++) {
            p[j*nx + nx-1] = p[j*nx + nx-2];
            p[j*nx + 0] = p[j*nx + 1];
        }
        // send p to rank + 1 (including rank = 0 and -1)
        send_to = (rank + 1) % size;
        if (n == 0 && it == 0) {
            MPI_Win_create(&p[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win0);
        }
        MPI_Win_fence(0, win0);
        MPI_Put(&p[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win0);
        MPI_Win_fence(0, win0);
        // send p to rank - 1 (including rank = 0 and -1)
        send_to = (rank - 1 + size) % size;
        if (n == 0 && it == 0) {
            MPI_Win_create(&p[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
        }
        MPI_Win_fence(0, win1);
        MPI_Put(&p[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win1);
        MPI_Win_fence(0, win1);
        if (rank == 0){ // fix values for rank = 0
            for (int i = 0; i < nx; i++) p[0*nx + i] = p[1*nx + i];
        } else if (rank == size-1){ // fix values for rank = -1
            for (int i = 0; i < nx; i++) p[(ny_split-1)*nx + i] = 0;
        }
    }
    // deepcopy
    vector<double> un = u;
    vector<double> vn = v;
    for (int j = 1; j < ny_split-1; j++) {
        for (int i = 1; i < nx-1; i++) { // loop order is already optimal
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
    for (int j = 1; j < ny_split-1; j++) {
        u[j*nx + 0]    = 0;
        u[j*nx + nx-1] = 0;
        v[j*nx + 0]    = 0;
        v[j*nx + nx-1] = 0;
    }
    // send u to rank + 1 (including rank = 0 and -1)
    send_to = (rank + 1) % size;
    if (n == 0) {
        MPI_Win_create(&u[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    }
    MPI_Win_fence(0, win2);
    MPI_Put(&u[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win2);
    MPI_Win_fence(0, win2);
    // send u to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    if (n == 0) {
        MPI_Win_create(&u[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win3);
    }
    MPI_Win_fence(0, win3);
    MPI_Put(&u[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win3);
    MPI_Win_fence(0, win3);
    // send v to rank + 1 (including rank = 0 and -1)
    send_to = (rank + 1) % size;
    if (n == 0) {
        MPI_Win_create(&v[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win4);
    }
    MPI_Win_fence(0, win4);
    MPI_Put(&v[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win4);
    MPI_Win_fence(0, win4);
    // send v to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    if (n == 0) {
        MPI_Win_create(&v[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win5);
    }
    MPI_Win_fence(0, win5);
    MPI_Put(&v[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win5);
    MPI_Win_fence(0, win5);
    if (rank == 0){
        for (int i = 0; i < nx; i++) {
            u[0*nx + i] = 0;
            v[0*nx + i] = 0;
        }
    } else if (rank == size-1){
        for (int i = 0; i < nx; i++) {
            u[(ny_split-1)*nx + i] = 1;
            v[(ny_split-1)*nx + i] = 0;
        }
    }
    MPI_Gatherv(&u[0], size-1, MPI_DOUBLE, &u0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&v[0], size-1, MPI_DOUBLE, &v0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&p[0], size-1, MPI_DOUBLE, &p0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&b[0], size-1, MPI_DOUBLE, &b0[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
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
    }
}
double toc = MPI_Wtime();
double time = toc - tic;
printf("%lf s",time);
MPI_Win_free(&win0);
MPI_Win_free(&win1);
MPI_Win_free(&win2);
MPI_Win_free(&win3);
MPI_Win_free(&win4);
MPI_Win_free(&win5);
MPI_Finalize();
} // close int main()

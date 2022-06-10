#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
using namespace std;
// python: 
// normal cpp: 
// openmp: 
// mpi: 
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
int size, rank;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//auto tic = chrono::steady_clock::now();
int nx = 41;
int ny = 41;
//int nt = 500;
int nt = 5; // debug
int nit = 50;
int send_to = 0;
int ny_split = 0;
int ny_splits[size];
int displacements[size];
double dx = 2 / (double(nx) - 1);
double dy = 2 / (double(ny) - 1);
double dt = 0.01;
double rho = 1;
double nu = 0.02;

// split j for MPI
// split j = 1 ~ ny-2 into size
displacements[0] = 0;
for (int i = 1; i < size; i++) {
    if (i != size-1) {
        ny_splits[i] = double(ny-2)/double(size); // nysplit for rank =/= size-1
    } else {
        ny_splits[i] = (ny-2) - (size-1)*double(ny-2)/double(size);
    }
    ny_splits[i] = ny_splits[i] + 2; // include before and after elements
    if (i == 0) { 
        displacements[0] = 0;
    } else {
        displacements[i] = displacements[i-1] + ny_splits[i-1];
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
printf("rank: %d,ny_split:%d,displacement:%d\n", rank, ny_split, displacements[rank]); // debug
MPI_Win win;
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
        MPI_Win_create(&p[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        MPI_Put(&p[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
        // send p to rank - 1 (including rank = 0 and -1)
        send_to = (rank - 1 + size) % size;
        MPI_Win_create(&p[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        MPI_Put(&p[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
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
    MPI_Win_create(&u[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(&u[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send u to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    MPI_Win_create(&u[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(&u[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send v to rank + 1 (including rank = 0 and -1)
    send_to = (rank + 1) % size;
    MPI_Win_create(&v[0*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(&v[(ny_split-2)*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send v to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    MPI_Win_create(&v[(ny_split-1)*nx], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(&v[1*nx], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
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
    if (rank == 0) {
        MPI_Gather(&u, ny_split*nx, MPI_DOUBLE, &u0, ny_split*nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&v, ny_split*nx, MPI_DOUBLE, &v0, ny_split*nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&p, ny_split*nx, MPI_DOUBLE, &p0, ny_split*nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(&u, ny_split*nx, MPI_DOUBLE, &u0, ny_splits, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&v, ny_split*nx, MPI_DOUBLE, &v0, ny_splits, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&p, ny_split*nx, MPI_DOUBLE, &p0, ny_splits, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
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
//auto toc = chrono::steady_clock::now();
//double time = chrono::duration<double>(toc - tic).count();
//printf("%lf s",time);
//MPI_Win_free(&win);
MPI_Finalize();
} // close int main()

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
using namespace std;
typedef vector<vector<double>> matrix;
// python: 
// normal cpp: 
// openmp: 
// mpi: 
int main(int argc, char** argv) {
auto tic = chrono::steady_clock::now();
int nx = 41;
int ny = 41;
int nt = 500;
//int nt = 5; // debug
int nit = 50;
int send_to = 0;
double dx = 2 / (double(nx) - 1);
double dy = 2 / (double(ny) - 1);
double dt = 0.01;
double rho = 1;
double nu = 0.02;
MPI_Init(&argc, &argv);
int size, rank;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// split j for MPI
// split j = 1 ~ ny-2 into size
if (rank == size-1) {
    int ny_split0 = (ny-2)/size; // nysplit for rank =/= size-1
    int ny_split = (ny-2) - (size-1)*ny_split0;
    ny_split = ny_split + 2; // include before and after elements
} else {
    int ny_split = (ny-2)/size;
    ny_split = ny_split + 2; // include before and after elements
}

// np.zeros() default dtype float64 = double (in c)
// vector defaults to zero
// store split data
matrix u(ny_split,vector<double>(nx));
matrix v(ny_split,vector<double>(nx));
matrix p(ny_split,vector<double>(nx));
matrix b(ny_split,vector<double>(nx));
// store all data
matrix u0(ny,vector<double>(nx));
matrix v0(ny,vector<double>(nx));
matrix p0(ny,vector<double>(nx));
printf("rank: %d,ny_split:%d\n", rank, ny_split); // debug
for (int n = 0; n < nt; n++) {
    MPI_Win win;
    for (int j = 1; j < ny_split-1; j++) {
        for (int i = 1; i < nx-1; i++) { // loop order is already optimal
            b[j][i] = rho * (
                1 / dt * ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                pow(((u[j][i+1] - u[j][i-1]) / (2 * dx)), 2) - 
                2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) - 
                pow(((v[j+1][i] - v[j-1][i]) / (2 * dy)), 2)
            );
        }
    }
    for (int it = 0; it < nit; it++) {
        matrix pn = p; // deepcopy
        for (int j = 1; j < ny_split-1; j++) {
            for (int i = 1; i < nx-1; i++) { // loop order is already optimal
                p[j][i] = (
                    pow(dy, 2) * (pn[j][i+1] + pn[j][i-1]) +
                    pow(dx, 2) * (pn[j+1][i] + pn[j-1][i]) -
                    b[j][i] * pow(dx, 2) * pow(dy, 2)
                ) / (2 * (pow(dx, 2) + pow(dy, 2)));
            }
        }
        for (int j = 1; j < ny_split-1; j++) {
            p[j][nx-1] = p[j][nx-2];
            p[j][0] = p[j][1];
        }
        // send p to rank + 1 (including rank = 0 and -1)
        send_to = (rank + 1) % size;
        MPI_Win_create(p[0], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        MPI_Put(p[ny_split-2], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
        // send p to rank - 1 (including rank = 0 and -1)
        send_to = (rank - 1 + size) % size;
        MPI_Win_create(p[ny_split-1], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        MPI_Put(p[1], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
        if (rank == 0){ // fix values for rank = 0
            for (int i = 0; i < nx; i++) p[0][i] = p[1][i];
        } else if (rank == size-1){ // fix values for rank = -1
            for (int i = 0; i < nx; i++) p[ny_split-1][i] = 0;
        }
    }
    // deepcopy
    matrix un = u;
    matrix vn = v;
    for (int j = 1; j < ny_split-1; j++) {
        for (int i = 1; i < nx-1; i++) { // loop order is already optimal
            u[j][i] = un[j][i] 
                - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                - un[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                + nu * dt / pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                + nu * dt / pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
            v[j][i] = vn[j][i] 
                - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                + nu * dt / pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                + nu * dt / pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
        }
    }
    for (int j = 1; j < ny_split-1; j++) {
        u[j][0]    = 0;
        u[j][nx-1] = 0;
        v[j][0]    = 0;
        v[j][nx-1] = 0;
    }
    // send u to rank + 1 (including rank = 0 and -1)
    send_to = (rank + 1) % size;
    MPI_Win_create(u[0], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(u[ny_split-2], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send u to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    MPI_Win_create(u[ny_split-1], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(u[1], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send v to rank + 1 (including rank = 0 and -1)
    send_to = (rank + 1) % size;
    MPI_Win_create(v[0], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(v[ny_split-2], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    // send v to rank - 1 (including rank = 0 and -1)
    send_to = (rank - 1 + size) % size;
    MPI_Win_create(v[ny_split-1], nx*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(v[1], nx, MPI_DOUBLE, send_to, 0, nx, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    if (rank == 0){
        for (int i = 0; i < nx; i++) {
            u[0][i] = 0;
            v[0][i] = 0;
        }
    } else if (rank == size-1){
        for (int i = 0; i < nx; i++) {
            u[ny_split-1][i] = 1;
            v[ny_split-1][i] = 0;
        }
    }
    if (rank == size-1){
        MPI_Gather(&u0[rank*ny_split0], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&v0[rank*ny_split0], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&p0[rank*ny_split0], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&u0[rank*ny_split], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&v0[rank*ny_split], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&p0[rank*ny_split], ny_split, MPI_DOUBLE, x, end-begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    /*
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
            mean_u += u[j][i]/(nx*ny);
            mean_v += v[j][i]/(nx*ny);
            mean_p += p[j][i]/(nx*ny);
            mean_b += b[j][i]/(nx*ny);
            mean2_u += pow(u[j][i],2)/(nx*ny);
            mean2_v += pow(v[j][i],2)/(nx*ny);
            mean2_p += pow(p[j][i],2)/(nx*ny);
            mean2_b += pow(b[j][i],2)/(nx*ny);
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
*/
}
MPI_Win_free(&win);
MPI_Finalize();
auto toc = chrono::steady_clock::now();
double time = chrono::duration<double>(toc - tic).count();
printf("%lf s",time);
} // close int main()

#include <iostream>
#include <vector>
using namespace std;

int main() {

int nx = 41;
int ny = 41;
int nt = 500;
int nit = 50;
double dx = 2 / (nx - 1);
double dy = 2 / (ny - 1);
double dt = 0.01;
double rho = 1;
double nu = 0.02;

// np.zeros() default dtype float64 = double (in c)
// vector defaults to zero
vector<double> u(ny*nx); 
vector<double> v(ny*nx);
vector<double> p(ny*nx);
vector<double> b(ny*nx);

for (int n = 0; n < nt; n++) {
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            b[j*nx + i] = rho * (
                1 / dt * ((u[j*nx + (i+1)] - u[j*nx + (i-1)]) / (2 * dx) + (v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy)) -
                ((u[j*nx + (i+1)] - u[j*nx + (i-1)]) / (2 * dx))**2 - 
                2 * ((u[(j+1)*nx + i] - u[(j-1)*nx + i]) / (2 * dy) * (v[j*nx + (i+1)] - v[j*nx + (i-1)]) / (2 * dx)) - 
                ((v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2 * dy))**2
            );
        }
    }
    for (int it = 0; it < nit; it++) {  
        vector<double> pn = p; // deepcopy
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[j*nx + i] = (
                    dy**2 * (pn[j*nx + (i+1)] + pn[j*nx + (i-1)]) +
                    dx**2 * (pn[(j+1)*nx + i] + pn[(j-1)*nx + i]) -
                    b[j*nx + i] * dx**2 * dy**2
                ) / (2 * (dx**2 + dy**2));
            }
        }
        for (int j = 1; j < ny-1; j++) {
            p[j*nx + nx-1] = p[j*nx + nx-2];
            p[j*nx + 0] = p[j*nx + 1];
        }
        for (int i = 1; i < nx-1; i++) {
            p[0*nx + i] = p[1*nx + i];
            p[(ny-1)*nx + i] = 0;
        }
    }
    // deepcopy
    vector<double> un = u;
    vector<double> vn = v;
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            u[j*nx + i] = 
                un[j*nx + i] - un[j*nx + i] * dt / dx * (un[j*nx + i] - un[j*nx + (i-1)])
                - un[j*nx + i] * dt / dy * (un[j*nx + i] - un[(j-1)*nx + i])
                - dt / (2 * rho * dx) * (p[j*nx + (i+1)] - p[j*nx + (i-1)])
                + nu * dt / dx**2 * (un[j*nx + (i+1)] - 2 * un[j*nx + i] + un[j*nx + (i-1)])
                + nu * dt / dy**2 * (un[(j+1)*nx + i] - 2 * un[j*nx + i] + un[(j-1)*nx + i]);
            v[j*nx + i] = 
                vn[j*nx + i] - vn[j*nx + i] * dt / dx * (vn[j*nx + i] - vn[j*nx + (i-1)])
                - vn[j*nx + i] * dt / dy * (vn[j*nx + i] - vn[(j-1)*nx + i])
                - dt / (2 * rho * dx) * (p[(j+1)*nx + i] - p[(j-1)*nx + i])
                + nu * dt / dx**2 * (vn[j*nx + (i+1)] - 2 * vn[j*nx + i] + vn[j*nx + (i-1)])
                + nu * dt / dy**2 * (vn[(j+1)*nx + i] - 2 * vn[j*nx + i] + vn[(j-1)*nx + i]);
        }
    }
    for (int j = 1; j < ny-1; j++) {
        u[j*nx + 0]  = 0;
        u[j*nx + (nx-1)] = 0;
        v[j*nx + 0]  = 0;
        v[j*nx + (nx-1)] = 0;
    }
    for (int i = 1; i < nx-1; i++) {
        u[0*nx + i]  = 0;
        u[(ny-1)*nx + i] = 1;
        v[0*nx + i]  = 0;
        v[(ny-1)*nx + i] = 0;
    }
}

} // close int main()

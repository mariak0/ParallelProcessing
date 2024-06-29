#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

#define MAXVARS     (250)  /* max # of variables       */
#define EPSMIN      (1E-6)  /* ending value of stepsize  */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);

/* global variables */
unsigned long funevals = 0;

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
        usleep(1);  /* do not remove, introduces some artificial work */

    return fv;
}


double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}


int main(int argc, char *argv[])
{
    
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* problem parameters */
    int nvars = 4;      /* number of variables (problem dimension) */
    int ntrials = 64;   /* number of trials */
    double lower[MAXVARS], upper[MAXVARS];  /* lower and upper bounds */

    /* mds parameters */
    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0;
    double theta = 0.25;
    double delta = 0.25;

    double startpt[MAXVARS], endpt[MAXVARS];  /* initial and final point of mds */
    double fx;  /* function value at the final point of mds */
    int nt, nf; /* number of iterations and function evaluations used by mds */

    /* information about the best point found by multistart */
    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    /* local variables */
    int trial, i;
    double t0, t1;

    /* initialization of lower and upper bounds of search space */
    for (i = 0; i < MAXVARS; i++) lower[i] = -2.0;  /* lower bound: -2.0 */
    for (i = 0; i < MAXVARS; i++) upper[i] = +2.0;  /* upper bound: +2.0 */

    t0 = get_wtime();

    /* how many trials each process will handle */
    int trials_per_proc = ntrials / num_procs;
    int remainder = ntrials % num_procs;

    /* Distribute trials among processes */
    int start_trial = rank * trials_per_proc;
    int end_trial = (rank == num_procs - 1) ? start_trial + trials_per_proc + remainder : start_trial + trials_per_proc;

    for (trial = start_trial; trial < end_trial; trial++) {
        srand48(trial + rank);  // Ensure different seeds across ranks

        /* starting guess for rosenbrock test function, search space in [-2, 2) */
        for (i = 0; i < nvars; i++) {
            startpt[i] = lower[i] + (upper[i]-lower[i])*drand48();
        }

        int term = -1;
        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter, mu, theta, delta,
            &nt, &nf, lower, upper, &term);

#if DEBUG
        printf("\n\n\nMDS %d USED %d ITERATIONS AND %d FUNCTION CALLS, AND RETURNED\n", trial, nt, nf);
        for (i = 0; i < nvars; i++)
            printf("x[%3d] = %15.7le \n", i, endpt[i]);

        printf("f(x) = %15.7le\n", fx);
#endif

        /* best solution */
        if (fx < best_fx) {
            best_trial = trial;
            best_nt = nt;
            best_nf = nf;
            best_fx = fx;
            for (i = 0; i < nvars; i++)
                best_pt[i] = endpt[i];
        }
    }

    /* results from all processes */
    double global_best_fx;
    int global_best_trial;
    MPI_Allreduce(&best_fx, &global_best_fx, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (best_fx == global_best_fx) {
        global_best_trial = best_trial;
        MPI_Bcast(best_pt, nvars, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        best_nt = nt;
        best_nf = nf;
    } else {
        MPI_Bcast(best_pt, nvars, MPI_DOUBLE, MPI_ROOT, MPI_COMM_WORLD);
    }

    t1 = get_wtime();

    if (rank == 0) {
        printf("\n\nFINAL RESULTS:\n");
        printf("Elapsed time = %.3lf s\n", t1-t0);
        printf("Total number of trials = %d\n", ntrials);
        printf("Best result at trial %d used %d iterations, %d function calls and returned\n", global_best_trial, best_nt, best_nf);
        for (i = 0; i < nvars; i++) {
            printf("x[%3d] = %15.7le \n", i, best_pt[i]);
        }
        printf("f(x) = %15.7le\n", global_best_fx);
    }

    MPI_Finalize();

    return 0;
}

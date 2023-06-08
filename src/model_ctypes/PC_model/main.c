#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>  

#include "./ina.h"

#define Ith(v,i)    NV_Ith_S(v,i)
#define IJth(A,i,j) SM_ELEMENT_D(A, i, j)
#define T0    RCONST(0.0) 
#define RTOL  RCONST(1.0e-13)
#define VTOL  RCONST(1.0e-12)
#define ATOL  RCONST(1.0e-8)

// static void PrintFinalStats(void *cvode_mem);
// static int check_retval(void *returnvalue, const char *funcname, int opt);

int rhs(realtype t, N_Vector y, N_Vector ydot, void *data) {
        double *C = (double *)data, *A = ((double *)data) + C_SIZE;
        SUNContext sunctx;
        int retval;
        retval = SUNContext_Create(NULL, &sunctx);
        N_Vector V_C = N_VMake_Serial(C_SIZE, C, sunctx);
        N_Vector V_A = N_VMake_Serial(A_SIZE, A, sunctx);
        compute_rates(t, y, V_C, V_A, ydot);
        N_VDestroy_Serial(V_A);
        N_VDestroy_Serial(V_C);
        SUNContext_Free(&sunctx);
        return 0;
}

int run(double *S, double *C,
        double *time_array, double *voltage_command_array, int array_length,
        double *output_S, double *output_A) {
        
        double data[C_SIZE + A_SIZE];
        double *A = data + C_SIZE;
        realtype t, tout, reltol;
        SUNContext sunctx;
        SUNMatrix Matrix;
        SUNLinearSolver LS;
        void *cvode_mem;
        int retval;

        for (int i = 0; i < C_SIZE; ++i) {
                data[i] = C[i];
        }
        retval = SUNContext_Create(NULL, &sunctx);
        N_Vector V_S = N_VMake_Serial(S_SIZE, S, sunctx);
        N_Vector V_C = N_VMake_Serial(C_SIZE, C, sunctx);
        N_Vector V_A = N_VMake_Serial(A_SIZE, A, sunctx);
        
        compute_algebraic(T0, V_S, V_C, V_A);
        memcpy(output_S, NV_DATA_S(V_S), S_SIZE * sizeof(double));
        memcpy(output_A, NV_DATA_S(V_A), A_SIZE * sizeof(double));
        
        // INIT
        cvode_mem = CVodeCreate(CV_BDF, sunctx);
        retval = CVodeInit(cvode_mem, rhs, T0, V_S);
        retval = CVodeSetUserData(cvode_mem, data);
        
        // TOLERANCES
        N_Vector abstol = N_VNew_Serial(S_SIZE, sunctx);
        Ith(abstol,0) = VTOL;//v_comp
        Ith(abstol,1) = VTOL;//v_p
        Ith(abstol,2) = VTOL;//v_m
        Ith(abstol,3) = ATOL;//m
        Ith(abstol,4) = ATOL;//h
        Ith(abstol,5) = ATOL;//j
        Ith(abstol,6) = VTOL;//I_out
        reltol = RTOL;
        retval = CVodeSVtolerances(cvode_mem, reltol, abstol);
        retval = CVodeSetErrFile(cvode_mem, NULL);
        Matrix = SUNDenseMatrix(S_SIZE, S_SIZE, sunctx);
        LS = SUNLinSol_Dense(V_S, Matrix, sunctx);
        retval = CVodeSetLinearSolver(cvode_mem, LS, Matrix);
        // retval = CVodeSetErrFile(NULL, NULL);
        retval = CVodeSetUserData(cvode_mem, data);
        
        for (int i = 1; i < array_length; i++) {
                tout = time_array[i];
                data[29] = voltage_command_array[i];
                retval = CVodeSetStopTime(cvode_mem, tout);
                retval = CVode(cvode_mem, tout, V_S, &t, CV_NORMAL);
                memcpy(output_S + i * S_SIZE, NV_DATA_S(V_S), S_SIZE * sizeof(double));
                memcpy(output_A + i * A_SIZE, NV_DATA_S(V_A), A_SIZE * sizeof(double));
        }
        // PrintFinalStats(cvode_mem);
        N_VDestroy_Serial(V_S);
        N_VDestroy_Serial(V_A);
        N_VDestroy_Serial(V_C);
        N_VDestroy_Serial(abstol);
        CVodeFree(&cvode_mem);
        SUNContext_Free(&sunctx);
        SUNLinSolFree(LS);
        SUNMatDestroy(Matrix);

        return 2;
}

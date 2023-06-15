// #ifndef _INA_CVODE_H_
// #define _INA_CVODE_H_

#define S_SIZE 7
#define C_SIZE 32
#define A_SIZE 12


void initialize_states_default(N_Vector STATES);
void compute_algebraic(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC);
void compute_rates(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC, N_Vector RATES);

// #endif

#include <math.h>
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <stdio.h>
#include "ina.h"

#define Ith(v,i)    NV_Ith_S(v,i)
#define NM          RCONST(1.0e-9)
#define RNM          RCONST(1.0e9)
#define min_time_step          RCONST(5.0e-5)

void initialize_states_default(N_Vector STATES){
  Ith(STATES,0) = -80;//v_comp
  Ith(STATES,1) = -80;//v_m
  Ith(STATES,2) = 0.;//m
  Ith(STATES,3) = 1.;//h
  Ith(STATES,4) = 1.;//j
  Ith(STATES,5) = 0;//I_out
}

void compute_algebraic(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC){
  // tau_m = tau_m_const +  1 / (a0_m * exp(v_m / s_m) + b0_m * exp(- v_m / delta_m))
  Ith(ALGEBRAIC,0) = Ith(CONSTANTS,10) + 1/(Ith(CONSTANTS,6) * exp(Ith(STATES,1)/Ith(CONSTANTS,9)) + Ith(CONSTANTS,7)*exp(- Ith(STATES,1)/Ith(CONSTANTS,8)));
  
  // tau_h =  tau_h_const +  1 / (a0_h * exp(-v_m / s_h) + b0_h * exp(v_m / delta_h))
  Ith(ALGEBRAIC,1) = Ith(CONSTANTS,15) + 1/(Ith(CONSTANTS,11) * exp(- Ith(STATES,1)/Ith(CONSTANTS,14)) + Ith(CONSTANTS,12)*exp(Ith(STATES,1)/Ith(CONSTANTS,13)));
  
  // tau_j = tau_j_const + 1 / (a0_j * exp(-v_m / s_j) + b0_j * exp(v_m / delta_j))
  Ith(ALGEBRAIC,2) = Ith(CONSTANTS,20) + 1/(Ith(CONSTANTS,16) * exp(- Ith(STATES,1)/Ith(CONSTANTS,19)) + Ith(CONSTANTS,17)*exp(Ith(STATES,1)/Ith(CONSTANTS,18)));
  
  // m_inf = 1 / (1 + exp(-(v_half_m + v_m) / k_m));
  Ith(ALGEBRAIC,3) = 1 / (1 + exp(-(Ith(CONSTANTS,2) + Ith(STATES,1))/Ith(CONSTANTS,4)));
  
  // h_inf = 1 / (1 + exp((v_half_h + v_m) / k_h));
  Ith(ALGEBRAIC,4) = 1 / (1 + exp((Ith(CONSTANTS,3) + Ith(STATES,1))/Ith(CONSTANTS,5)));
  

  if (Ith(CONSTANTS,25)*Ith(CONSTANTS,26) <= min_time_step){
	// v_cp = v_c
        Ith(ALGEBRAIC,5) = Ith(CONSTANTS,0);
	// I_comp = 0
        Ith(ALGEBRAIC,9) = 0;
  }
  else{
  	// v_cp =  v_c + (v_c - v_comp)*(1/(1-alpha) - 1); 
  	Ith(ALGEBRAIC,5) = Ith(CONSTANTS,0) + (Ith(CONSTANTS,0) - Ith(STATES,0))*(1/(1 - Ith(CONSTANTS,28)) - 1);
  	// I_comp = 1e9 * x_c_comp * d v_comp / dt
  	Ith(ALGEBRAIC,9) = RNM * (Ith(CONSTANTS,0) - Ith(STATES,0))/(Ith(CONSTANTS,26)*(1 - Ith(CONSTANTS,28)));
  }

  // I_Na = g_max * h * pow(m,3) * (v_m - v_rev) * j ;
  Ith(ALGEBRAIC,7) = Ith(CONSTANTS,23) * Ith(STATES,3) * pow(Ith(STATES,2),3) * Ith(STATES,4)* (Ith(STATES,1) - Ith(CONSTANTS,1));
  
  // I_leak = g_leak * v_m;
  Ith(ALGEBRAIC,6) = Ith(CONSTANTS,24) * Ith(STATES,1);
  // I_c = 1e9 * c_m * dv_m / dt
  Ith(ALGEBRAIC,8) = RNM * ((Ith(ALGEBRAIC,5)  - Ith(STATES,1))/Ith(CONSTANTS,22) - NM*(Ith(ALGEBRAIC,6) + Ith(ALGEBRAIC,7)));
  // I_in = I_Na + I_leak +I_c - I_comp 
  Ith(ALGEBRAIC,10) = Ith(ALGEBRAIC,6) + Ith(ALGEBRAIC,7) + Ith(ALGEBRAIC,8) - Ith(ALGEBRAIC,9);
}

void compute_rates(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC, N_Vector RATES){

  compute_algebraic(time, STATES, CONSTANTS, ALGEBRAIC);
  if (Ith(CONSTANTS,25)*Ith(CONSTANTS,26) <= min_time_step){
	  Ith(RATES,0) = 0;
  } 
  else{
	  // v_comp = (v_c - v_comp) / (x_r_comp *  x_c_comp * (1 - alpha))
  	  Ith(RATES,0) = (Ith(CONSTANTS,0) - Ith(STATES,0))/(Ith(CONSTANTS,25) * Ith(CONSTANTS,26)*(1 - Ith(CONSTANTS,28)));
  }
  
  // v_m = (v_cp - v_m ) / (r_m * c_m) - 1e-9 * (I_Na + I_leak) / c_m ;
  Ith(RATES,1) = (Ith(ALGEBRAIC,5) - Ith(STATES,1))/(Ith(CONSTANTS,22)*Ith(CONSTANTS,21)) - NM*(Ith(ALGEBRAIC,6) + Ith(ALGEBRAIC,7))/Ith(CONSTANTS,21);
   
  // m = (m_inf - m) / tau_m
  Ith(RATES,2) = (Ith(ALGEBRAIC,3) - Ith(STATES,2))/Ith(ALGEBRAIC,0);

  // h = (h_inf - h) / tau_h
  Ith(RATES,3) = (Ith(ALGEBRAIC,4) - Ith(STATES,3))/Ith(ALGEBRAIC,1);

  // j = (h_inf - j) / tau_j
  Ith(RATES,4) = (Ith(ALGEBRAIC,4) - Ith(STATES,4))/Ith(ALGEBRAIC,2);
  
  // I_out = (I_in - I_out) / tau_z
  Ith(RATES,5) = (Ith(ALGEBRAIC,10) - Ith(STATES,5))/Ith(CONSTANTS,27);
  }

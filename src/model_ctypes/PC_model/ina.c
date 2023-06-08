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
  // double v_comp = Ith(STATES,0);
  Ith(STATES,0) = -80;//v_comp
  Ith(STATES,1) = -80;//v_p
  Ith(STATES,2) = -80;//v_m
  Ith(STATES,3) = 0.;//m
  Ith(STATES,4) = 1.;//h
  Ith(STATES,5) = 1.;//j
  Ith(STATES,6) = 0;//I_out
}

void compute_algebraic(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC){
  // tau_m = tau_m_const +  1 / (a0_m * exp(v_m / s_m) + b0_m * exp(- v_m / delta_m))
  Ith(ALGEBRAIC,0) = Ith(CONSTANTS,30) + 1/(Ith(CONSTANTS,2) * exp(Ith(STATES,2)/Ith(CONSTANTS,5)) + Ith(CONSTANTS,3)*exp(- Ith(STATES,2)/Ith(CONSTANTS,4)));
  
  // tau_h =  tau_h_const +  1 / (a0_h * exp(-v_m / s_h) + b0_h * exp(v_m / delta_h))
  Ith(ALGEBRAIC,1) = Ith(CONSTANTS,31) + 1/(Ith(CONSTANTS,6) * exp(- Ith(STATES,2)/Ith(CONSTANTS,9)) + Ith(CONSTANTS,7)*exp(Ith(STATES,2)/Ith(CONSTANTS,8)));
  
  // tau_j = tau_j_const + 1 / (a0_j * exp(-v_m / s_j) + b0_j * exp(v_m / delta_j))
  Ith(ALGEBRAIC,2) = Ith(CONSTANTS,14) + 1/(Ith(CONSTANTS,10) * exp(- Ith(STATES,2)/Ith(CONSTANTS,13)) + Ith(CONSTANTS,11)*exp(Ith(STATES,2)/Ith(CONSTANTS,12)));
  
  // m_inf = 1 / (1 + exp(-(v_half_m + v_m) / k_m));
  Ith(ALGEBRAIC,3) = 1 / (1 + exp(-(Ith(CONSTANTS,20) + Ith(STATES,2))/Ith(CONSTANTS,22)));
  
  // h_inf = 1 / (1 + exp((v_half_h + v_m) / k_h));
  Ith(ALGEBRAIC,4) = 1 / (1 + exp((Ith(CONSTANTS,21) + Ith(STATES,2))/Ith(CONSTANTS,23)));
  
  if (Ith(CONSTANTS,24)*Ith(CONSTANTS,25) <= min_time_step){
	// v_cp = v_c
        Ith(ALGEBRAIC,5) = Ith(CONSTANTS,29);
	// I_comp = 0
        Ith(ALGEBRAIC,10) = 0;
  }
  else{
  	// v_cp =  v_c + (v_c - v_comp)*(1/(1-alpha) - 1); 
  	Ith(ALGEBRAIC,5) = Ith(CONSTANTS,29) + (Ith(CONSTANTS,29) - Ith(STATES,0))*(1/(1 - Ith(CONSTANTS,26)) - 1);
  	// I_comp = 1e9 * x_c_comp * d v_comp / dt
  	Ith(ALGEBRAIC,10) = RNM * (Ith(CONSTANTS,29) - Ith(STATES,0))/(Ith(CONSTANTS,25)*(1 - Ith(CONSTANTS,26)));
  }





  // I_Na = g_max * h * pow(m,3) * (v_m - v_rev) * j ;
  Ith(ALGEBRAIC,7) = Ith(CONSTANTS,17) * Ith(STATES,4) * pow(Ith(STATES,3),3) * Ith(STATES,5)* (Ith(STATES,2) - Ith(CONSTANTS,28));
  
  //*******************************
  // [IN] 
  // I_leak = g_leak * v_m;
  Ith(ALGEBRAIC,6) = Ith(CONSTANTS,18) * Ith(STATES,2);
  // I_c = 1e9 * c_m * dv_m / dt
  Ith(ALGEBRAIC,8) = RNM * ((Ith(STATES,1) + Ith(CONSTANTS,27) - Ith(STATES,2))/Ith(CONSTANTS,15) - NM*(Ith(ALGEBRAIC,6) + Ith(ALGEBRAIC,7)));
  // I_p = 1e9 * c_p * dv_p / dt
  Ith(ALGEBRAIC,9) = RNM * ((Ith(ALGEBRAIC,5) - Ith(STATES,1))/Ith(CONSTANTS,16) + (Ith(STATES,2) - Ith(STATES,1) - Ith(CONSTANTS,27))/Ith(CONSTANTS,15));
  
  // [OUT] 
  // // I_leak = g_leak * v_p;
  // Ith(ALGEBRAIC,6) = Ith(CONSTANTS,18) * Ith(STATES,1);
  // // I_c = 1e9 * c_m * dv_m / dt
  // Ith(ALGEBRAIC,8) = RNM * Ith(CONSTANTS,1) * ((Ith(STATES,1) + Ith(CONSTANTS,27) - Ith(STATES,2))/(Ith(CONSTANTS,15)*Ith(CONSTANTS,1)) - NM*(Ith(ALGEBRAIC,7))/Ith(CONSTANTS,1));  
  // // I_p = 1e9 * c_p * dv_p / dt
  // Ith(ALGEBRAIC,9) = RNM * Ith(CONSTANTS,0) *((Ith(ALGEBRAIC,5) - Ith(STATES,1))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,16)) + (Ith(STATES,2) - Ith(STATES,1) - Ith(CONSTANTS,27))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,15)) - NM*Ith(ALGEBRAIC,6)/Ith(CONSTANTS,0));
  //*******************************

  // I_comp = 1e9 * x_c_comp * d v_comp / dt
  // Ith(ALGEBRAIC,10) = RNM * Ith(CONSTANTS,24) * (Ith(CONSTANTS,29) - Ith(STATES,0))/(Ith(CONSTANTS,24)*Ith(CONSTANTS,25)*(1 - Ith(CONSTANTS,26)));// Ith(RATES,0)
  
  // I_in = 1e9 *(v_cp - v_p)/r_p - I_comp 
  Ith(ALGEBRAIC,11) = RNM*(Ith(ALGEBRAIC,5) - Ith(STATES,1))/(Ith(CONSTANTS,16)) - Ith(ALGEBRAIC,10);
}

void compute_rates(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC, N_Vector RATES){

  compute_algebraic(time, STATES, CONSTANTS, ALGEBRAIC);
  if (Ith(CONSTANTS,24)*Ith(CONSTANTS,25) <= min_time_step){
	  Ith(RATES,0) = 0;
  } 
  else{
	  // v_comp = (v_c - v_comp) / (x_r_comp *  x_c_comp * (1 - alpha))
  	  Ith(RATES,0) = (Ith(CONSTANTS,29) - Ith(STATES,0))/(Ith(CONSTANTS,24) * Ith(CONSTANTS,25)*(1 - Ith(CONSTANTS,26)));
  }
  
  //*******************************
  // [IN]
  // v_p = (v_cp  - v_p) / (r_p * c_p) + (v_m - v_p - v_off) / (R * c_p) 
  Ith(RATES,1) = (Ith(ALGEBRAIC,5) - Ith(STATES,1))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,16)) + (Ith(STATES,2) - Ith(STATES,1) - Ith(CONSTANTS,27))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,15));
  // v_m = (v_p + v_off - v_m ) / (r_m * c_m) - 1e-9 * (I_Na + I_leak) / c_m ;
  Ith(RATES,2) = (Ith(STATES,1) + Ith(CONSTANTS,27) - Ith(STATES,2))/(Ith(CONSTANTS,15)*Ith(CONSTANTS,1)) - NM*(Ith(ALGEBRAIC,6) + Ith(ALGEBRAIC,7))/Ith(CONSTANTS,1);
  
  // [OUT]   
  // // v_p = (v_cp  - v_p) / (r_p * c_p) + (v_m - v_p - v_off) / (R * c_p)) - 1e-9 * I_leak/c_p
  // Ith(RATES,1) = (Ith(ALGEBRAIC,5) - Ith(STATES,1))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,16)) + (Ith(STATES,2) - Ith(STATES,1) - Ith(CONSTANTS,27))/(Ith(CONSTANTS,0)*Ith(CONSTANTS,15)) - NM*Ith(ALGEBRAIC,6)/Ith(CONSTANTS,0) ;
  // // v_m = (v_p + v_off - v_m ) / (r_m * c_m) - 1e-9 * (I_Na ) / c_m ;
  // Ith(RATES,2) = (Ith(STATES,1) + Ith(CONSTANTS,27) - Ith(STATES,2))/(Ith(CONSTANTS,15)*Ith(CONSTANTS,1)) - NM*(Ith(ALGEBRAIC,7))/Ith(CONSTANTS,1);
  //*******************************
  
  // m = (m_inf - m) / tau_m
  Ith(RATES,3) = (Ith(ALGEBRAIC,3) - Ith(STATES,3))/Ith(ALGEBRAIC,0);

  // h = (h_inf - h) / tau_h
  Ith(RATES,4) = (Ith(ALGEBRAIC,4) - Ith(STATES,4))/Ith(ALGEBRAIC,1);

  // j = (h_inf - j) / tau_j
  Ith(RATES,5) = (Ith(ALGEBRAIC,4) - Ith(STATES,5))/Ith(ALGEBRAIC,2);
  
  // I_out = (I_in - I_out) / tau_z
  Ith(RATES,6) = (Ith(ALGEBRAIC,11) - Ith(STATES,6))/Ith(CONSTANTS,19);
  }


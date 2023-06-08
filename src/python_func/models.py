import numpy as np
def calculate_tau_m_from_ab(v_m, params):
    tau_m = params.tau_m_const + 1 / (params.a0_m * np.exp(v_m / params.s_m) + params.b0_m * np.exp(- v_m / params.delta_m))
    return tau_m
def calculate_tau_h_from_ab(v_m, params):
    tau_h = params.tau_h_const + 1 / (params.a0_h * np.exp(-v_m / params.s_h) + params.b0_h * np.exp(v_m / params.delta_h))
    return tau_h
def calculate_tau_j_from_ab(v_m, params):
    tau_j = params.tau_j_const + 1 / (params.a0_j * np.exp(-v_m / params.s_j) + params.b0_j * np.exp(v_m / params.delta_j))
    return tau_j

def calculate_m_inf_from_ab(v_m, params):
    m_inf = 1/( 1 + np.exp(-v_m*(1/params.delta_m + 1/params.s_m))*params.b0_m/params.a0_m)
    return m_inf

def calculate_h_inf_from_ab(v_m, params):
    h_inf = 1 / (1 + np.exp(v_m*(1/params.delta_h + 1/params.s_h))*params.b0_h/params.a0_h )
    return h_inf

def calculate_m_inf_from_v_half(v_m, params):
    m_inf =  1 / (1 + np.exp(-(params.v_half_m + v_m) / params.k_m))
    return m_inf
def calculate_h_inf_from_v_half(v_m, params):
    h_inf = 1 / (1 + np.exp((params.v_half_h + v_m) / params.k_h))
    return h_inf
def calculate_tau_m_from_v_half(v_m, params):
    tau_m = params.tau_m_const + 1 / ((1 + np.exp(-(params.v_half_m + v_m) / params.k_m))*params.a0_m * np.exp(v_m / params.s_m))
    return tau_m
def calculate_tau_h_from_v_half(v_m, params):
    tau_h = params.tau_h_const + 1 / ((1 + np.exp((params.v_half_h + v_m) / params.k_h))*params.a0_h * np.exp(-v_m / params.s_h))
    return tau_h
def calculate_tau_j_from_v_half(v_m, params):
    tau_j = params.tau_j_const + 1 / ((1 + np.exp((params.v_half_h + v_m) / params.k_h))*params.a0_j * np.exp(-v_m / params.s_j))
    return tau_j

def model1(v_m,params):
    m_inf = calculate_m_inf_from_v_half(v_m, params)
    h_inf = calculate_h_inf_from_v_half(v_m, params)
    tau_m = calculate_tau_m_from_ab(v_m, params)
    tau_h = calculate_tau_h_from_ab(v_m, params)
    tau_j = calculate_tau_j_from_ab(v_m, params)
    return {'tau_m': tau_m,
            'tau_h': tau_h,
            'tau_j': tau_j,
            'm': m_inf,
            'h': h_inf}

def model2(v_m,params):
    m_inf = calculate_m_inf_from_ab(v_m, params)
    h_inf = calculate_h_inf_from_ab(v_m, params)
    tau_m = calculate_tau_m_from_ab(v_m, params)
    tau_h = calculate_tau_h_from_ab(v_m, params)
    tau_j = calculate_tau_j_from_ab(v_m, params)
    return {'tau_m': tau_m,
            'tau_h': tau_h,
            'tau_j': tau_j,
            'm': m_inf,
            'h': h_inf}

def model3(v_m,params):
    m_inf = calculate_m_inf_from_v_half(v_m, params)
    h_inf = calculate_h_inf_from_v_half(v_m, params)
    tau_m = calculate_tau_m_from_v_half(v_m, params)
    tau_h = calculate_tau_h_from_v_half(v_m, params)
    tau_j = calculate_tau_j_from_v_half(v_m, params)
    return {'tau_m': tau_m,
            'tau_h': tau_h,
            'tau_j': tau_j,
            'm': m_inf,
            'h': h_inf}


def model4(v_m,params):
    params['tau_m_const'] = 0
    params['tau_h_const'] = 0
    m_inf = calculate_m_inf_from_v_half(v_m, params)
    h_inf = calculate_h_inf_from_v_half(v_m, params)
    tau_m = calculate_tau_m_from_ab(v_m, params)
    tau_h = calculate_tau_h_from_ab(v_m, params)
    tau_j = calculate_tau_j_from_ab(v_m, params)
    return {'tau_m': tau_m,
            'tau_h': tau_h,
            'tau_j': tau_j,
            'm': m_inf,
            'h': h_inf}



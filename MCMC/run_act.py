import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pymc as pm
import pymcmcstat.ParallelMCMC
import pytensor.tensor as at
from pytensor.tensor.basic import as_tensor_variable

sys.path.append('../ga/mpi_scripts/')
from classes import DRAM, CustomLogLike
from functions import func_model, return_data_cut
from gene_utils import update_C_from_genes
from ina_model import InaModel
from io_utils import collect_results
from copy import deepcopy

trace_name = 'activation#1'

ga_result_dirname = '../results/ga/' #dir with the results of GA optimization
dirname_case = os.path.join(ga_result_dirname, trace_name)
case = '230709_115003'#os.listdir(dirname_case)[0]



#Create_output_dir
path_mcmc_output = f'../results/pymc/activation/{trace_name}'

abspath_mcmc_output = os.path.abspath(path_mcmc_output)
path_mcmc_output_list = abspath_mcmc_output.split('/')
for k in range(2, len(path_mcmc_output_list)+1):
    part_path = os.path.join('/',*abspath_mcmc_output.split('/')[:k])
    logic = os.path.isdir(part_path)
    if not logic:
        os.mkdir(part_path)

time_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
full_filename = os.path.join(abspath_mcmc_output, time_suffix+'.nc')

### For using gradient of trace, instead current trace: GRAD=True
### For log parameters: LOG=True

INACT = 0       #Activation/Inactivation protocol
GRAD = 1        #Derivative of trace is used
LOG = 1         #Logarithmic scale
PIPETTE = 0     #Model accounting for pipette

#nchain = 40
#ndraws = 10_000
nchain = 4      #number of MCMC chains
ndraws = 100    #number of draws
nburn = 0       #burn-in period. Set to 0, we remove it manually.

print(f'Trace name = {trace_name}')
print(f'GRAD = {GRAD}\nLOG = {LOG}\nPIPETTE = {PIPETTE}\nDraws = {ndraws}\nChains = {nchain}')


result = collect_results(case, dirname_case, dump_keys=['best'])
sol_best = result['sol_best']
config = result['config']
bounds = config['runtime']['bounds']
phenotype_best = result['phenotype_best']['individual']
data = config['experimental_conditions']['individual']['phenotype']

if PIPETTE:
    model_dir = '../src/model_ctypes/PC_model_with_pipette/'
else:
    model_dir = '../src/model_ctypes/PC_model'

legend_states = config['runtime']['legend']['states'] 
legend_algebraic =  config['runtime']['legend']['algebraic'] 
legend_constants =  config['runtime']['legend']['constants']

A = deepcopy(legend_algebraic)
C = deepcopy(legend_constants)
S = deepcopy(legend_states)


filename_so = config['filename_so'] 
Ina = InaModel(filename_so)

data_no_grad = data.copy()

weight = config['experimental_conditions']['individual']['sample_weight']
weight_grad = config['experimental_conditions']['individual']['sample_derivative_weight']


#Parameters that we do not want to sample
pass_params = ['alpha',
            #    'c_m', 'R', 'g_max',
            #    'x_c_comp', 'x_r_comp', 'tau_z',
            #    'a0_m', 'b0_m', 's_m', 'delta_m', 'tau_m_const',
            #    'a0_h', 'b0_h', 's_h', 'delta_h', 'tau_h_const',
            #    'a0_j', 'b0_j', 's_j', 'delta_j', 'tau_j_const',
            #    'v_half_m', 'k_m',
            #    'v_half_h', 'k_h',
            #    'v_rev', 'g_leak',
               ]

experiment_condition_name = list(config['experimental_conditions'].keys())[-1]
update_C_from_genes(C, sol_best, experiment_condition_name, config)

const_from_sol_best = deepcopy(C)
for param in const_from_sol_best.index:
    if param not in pass_params:
        const_from_sol_best[param] = legend_constants[param]

# delete pass parameters from m_index and sol_best_before by their number
m_index = config['runtime']['m_index']
mask_multipliers = config['runtime']['mask_multipliers']

delete_parameters = []
for param in pass_params:
    for condition_name in config['experimental_conditions']:
        if param in config['experimental_conditions'][condition_name]['params']:
            delete_index = (condition_name, param)
            param_number = np.where(m_index == delete_index)   
            delete_parameters.append(param_number)

m_index = m_index.delete(delete_parameters)
sol_best_before = np.delete(sol_best.values, delete_parameters)
bounds = np.delete(bounds, delete_parameters, 0)
if LOG:
    mask_log = np.delete(mask_multipliers, delete_parameters)    
    for i, logic in enumerate(mask_log):
        if logic:
            bounds[i] = np.log10(bounds[i])
            sol_best_before[i] = np.log10(sol_best_before[i])
else:
    mask_log = None 


change_bounds = {'v_half_h':np.array([30., 100.]),
                  'k_h':np.array([3., 20.]),
                  'v_half_m':np.array([-10., 60.]),
                  'k_m':np.array([3., 20.]),
                  'g_leak':np.array([0.01, 20.]),
                  'g_max':np.array([0.005, 20.]),
                  'v_rev':np.array([1., 150.])}

for param in change_bounds:
    if param not in pass_params:
        param_ind = np.where(m_index.get_level_values(1)==param)[0][0]
        bounds[param_ind] = change_bounds[param]

if GRAD:
    flag = 'grad'
    data = np.gradient(data)
else:
    flag = 'ina'


matrix_name = 'proposal_all_parameters.csv'
param_cov_mat = pd.read_csv(matrix_name, index_col=0)
cov_mat_len = len(param_cov_mat)
param_cov_mat = param_cov_mat.loc[list(m_index.get_level_values(1)), 
                                  list(m_index.get_level_values(1))]
param_cov_mat = param_cov_mat.values * len(sol_best_before) / cov_mat_len

#Some parameters are multipliers. The routine below converts them to absolute values.


trace = func_model(sol_best_before,
                   m_index,
                   Ina=Ina,
                   const=const_from_sol_best,
                   config=config,
                   flag=flag,
                   mask_log=mask_log,)

#Calculate the solution with no sodium current at all. 
#Used to check for 'bad' initial MCMC states.
sol_no_ina = sol_best_before.copy()
sol_no_ina[np.where(m_index.get_level_values(1) == 'g_max')[0][0]] = -100
no_ina = func_model(sol_no_ina, 
                    m_index, 
                    Ina=Ina, 
                    const=const_from_sol_best, 
                    config=config, 
                    flag=flag,
                    mask_log=mask_log,
                    ) 

downsampl = 10  #downsampling    
    
weight_cut = weight.drop('t', axis = 1)
mask_cut = np.zeros_like(weight_cut)

mask_cut_big = mask_cut.copy()
mask_cut_big[(np.where(weight_cut > 1.)[0][::1], 
              np.where(weight_cut > 1.)[1][::1])]= 1.
mask_cut_big[:, 2300:] = 0
mask_cut_big = mask_cut_big.reshape(-1)
mask_cut_big = mask_cut_big.astype('bool')


mask_cut[(np.where(weight_cut > 1.)[0][::downsampl], 
              np.where(weight_cut > 1.)[1][::downsampl])]= 1.
mask_cut[:, 2300:] = 0
mask_cut[1::2] = 0
mask_cut[:6] = 0

mask_cut_down = mask_cut.reshape(-1)
mask_cut_down = mask_cut_down.astype('bool')

data_cut_size = np.sum(mask_cut_down)
data_cut = np.zeros_like(data_cut_size)
delta_data = np.array(data - trace)
data_cut = delta_data[mask_cut_down.astype(bool)]
       

diff_no_ina = np.sum((data[mask_cut_down] - no_ina[mask_cut_down])**2)
diff_best = np.sum((data[mask_cut_down] - trace[mask_cut_down])**2)

ina_dict = {}
ina_dict['data'] = data
ina_dict['m_index'] = m_index
ina_dict['Ina'] = Ina
ina_dict['const'] = const_from_sol_best
ina_dict['config'] = config
ina_dict['mask_log'] = mask_log
ina_dict['mask_cut'] = mask_cut_down
ina_dict['flag'] = flag
ina_dict['mask_cut_big'] = mask_cut_big
ina_dict['return_data_cut'] = return_data_cut


sols = []
sols.append(sol_best_before)

len_random_start = nchain - len(sols)
if nchain - len(sols) < 0:
    len_random_start = 0


initial_values_parameters = []


initial_bounds = {'v_half_m':np.array([15.0, 40.]),
                  'v_half_h':np.array([60., 90.]),
                  'k_m':np.array([5., 15.]),
                  'k_h':np.array([5., 15.])}

max_cmodel_time = 1.3
min_dv_between_act_inact = 30

while len(initial_values_parameters)!=len_random_start:
    if True:
        s_b_b = sols[0] #Solution of GA is included as initial state of one of the chains.
#!!!Following piece of code was used to reduce burn-in period
#!!!Several GA solutions (three_starts) were used as initial Markov Chain states.
#!!!Hardcoded magic number 13 is 40 chains divided by 3 initial solutions.
#    if len(initial_values_parameters)%13 == 0 :
#        s_b_b = sols[len(initial_values_parameters)//13] 
        
        up, low = 0.8, 1.2
        new_bounds = []

        for i, [index, param_bounds] in enumerate(zip(m_index.get_level_values(1), bounds)):
            lb, ub = np.sort([s_b_b[i]*up, s_b_b[i]*low])
            if lb < param_bounds[0]:
                lb = param_bounds[0]*1.01
            if ub > param_bounds[1]:
                ub = param_bounds[1]*0.99

            new_bounds.append([lb, ub])
        new_bounds = np.array(new_bounds)

        for param in initial_bounds:
            if param not in pass_params:
                param_ind = np.where(m_index.get_level_values(1) == param)[0][0]
                new_bounds[param_ind] = initial_bounds[param]

#Initial states of chain are generated below within new_bounds.
#For some reason we use pymcmcstat here. This is a leftover from previous version, but it works. 
    initial_value = pymcmcstat.ParallelMCMC.generate_initial_values(1,
                                                                    len(new_bounds),
                                                                    new_bounds.T[0],
                                                                    new_bounds.T[1])
    
#This is to decrease burn-in. We don't want to start with 'bad' solutions with no Ina. 
    check_time = -time.time()
    trace_now = func_model(initial_value[0],
                           m_index,
                           Ina=Ina,
                           const=const_from_sol_best,
                           config=config,
                           flag=flag,
                           mask_log=mask_log,
                           )
    check_time += time.time()
    diff_trace_now = np.sum((data[mask_cut_down] - trace_now[mask_cut_down])**2)

    if np.all([_ not in pass_params for _ in ['v_half_h', 'v_half_m']]):
        v_half_h_ind = np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]
        v_half_m_ind = np.where(m_index.get_level_values(1) == 'v_half_m')[0][0]
        diff_v = initial_value[0][v_half_h_ind] - initial_value[0][v_half_m_ind]
    else:
        diff_v = min_dv_between_act_inact

    if np.all([check_time < max_cmodel_time, #time to model counting is not too much
               diff_v >= min_dv_between_act_inact, #start difference between half activationd and half inactivation is more than min_dv_between_act_inact
               diff_trace_now < diff_no_ina]): #trace from start parameters not equal trace without sodium curreny
        initial_values_parameters.append(initial_value[0])

initial_values_parameters = np.array(initial_values_parameters)
initial_values_parameters = np.append(initial_values_parameters, sols, axis=0)


start_dicts = []
for sol in initial_values_parameters:
    start_val = np.array(sol)
    start_vals = {}
    start_vals['parameters'] = start_val
    start_dicts.append(start_vals)
start_dicts = np.array(start_dicts)
# print(f'start_dicts = {start_dicts}')

model = pm.Model()
transform = None
scale = 1   #Proposal matrix is empirical covariance matrix multiplied by scale factor.
tune_interval = 10_000  #Proposal matrix is adaptive change it every N steps. 10_000 if initial proposal is good enough.
scale_factor = 1/5  #scaling factor used for delayed rejection

n = 1
p = np.shape(data_cut)[0]
S_0 = np.eye(p) * data_cut**2


mu = (bounds.T[0] + bounds.T[1]) / 2
special_mu = {'v_half_h':76.,
              'k_h':11,
              'v_rev':18}

for param in special_mu:
    if param not in pass_params:
        mu[np.where(m_index.get_level_values(1)==param)[0][0]] = special_mu[param]


sigma = (bounds.T[1] - bounds.T[0]) * 2
special_sigma = {'v_half_h':9.,
              'k_h':3,
              'v_rev':7}
for param in special_sigma:
    if param not in pass_params:
        sigma[np.where(m_index.get_level_values(1)==param)[0][0]] = special_sigma[param]


with model:
    parameters = pm.Normal('parameters',
                           mu=mu,
                           sigma=sigma,
                           transform=transform)


    loglike = CustomLogLike(ina_dict=ina_dict,
                            n=n,
                            p=p,
                            S=S_0,
                            )

    params = as_tensor_variable(parameters,)
    pm.Potential("likelihood", loglike(params))

    step_parameters = DRAM([parameters],
                           loglike=loglike,
                           S=param_cov_mat,
                           proposal_dist=pm.MultivariateNormalProposal,
                           scaling=scale,
                           tune='S',
                           tune_interval=tune_interval,
                           bounds=bounds.T,
                           initial_values_size=len(sol_best_before),
                           transform=transform,
                           scale_factor=scale_factor,
                           )


    steps = step_parameters

idata = pm.sample(ndraws,
                  tune=nburn,
                  step=steps,
                  model = model,
                  chains=nchain,
                  cores=nchain,
                  return_inferencedata=True,
                  initvals=start_dicts,
                 )



idata.to_netcdf(full_filename)
print(f'MCMC result saved at {full_filename}')
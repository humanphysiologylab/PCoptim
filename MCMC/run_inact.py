import os
import sys
import pytensor.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
import pymcmcstat.ParallelMCMC
import time

from pytensor.tensor.basic import as_tensor_variable
from datetime import datetime
sys.path.append('../ga/mpi_scripts/')
from ina_model import InaModel
from io_utils import collect_results
from functions import return_data_cut

from functions import func_model
from classes import CustomLogLike, DRAM


    
trace_name = 'inactivation'

### For using gradient of trace, instead current trace: GRAD=True
### For log parameters: LOG=True
print(trace_name)
INACT = 1       #Activation/Inactivation protocol
GRAD = 1        #Derivative of trace is used
LOG = 1         #Logarithmic scale
PIPETTE = 0     #Model accounting for pipette

#nchain = 40
#ndraws = 10_000
nchain = 4      #number of MCMC chains
ndraws = 500    #number of draws
nburn = 0       #burn-in period. Set to 0, we remove it manually.

print(f'Trace name = {trace_name}')
print(f'GRAD = {GRAD},\n LOG = {LOG},\n PIPETTE = {PIPETTE}')
print(f'Draws = {ndraws},\n Chains = {nchain}')

dirname = '../results/' #dir with the results of GA optimization
dirname_case = os.path.join(dirname, trace_name)
case = os.listdir(dirname_case)[0]

result = collect_results(case, dirname_case, dump_keys=['best'])
sol_best = result['sol_best']
config = result['config']
bounds = config['runtime']['bounds']
#print(config['experimental_conditions']['individual'].keys())
phenotype_best = result['phenotype_best']['individual']
data=config['experimental_conditions']['individual']['phenotype']

if PIPETTE:
    model_dir = '../src/model_ctypes/PC_model_with_pipette/'
else:
    model_dir = '../src/model_ctypes/PC_model'

legend_states = config['runtime']['legend']['states'] 
legend_algebraic =  config['runtime']['legend']['algebraic'] 
legend_constants =  config['runtime']['legend']['constants']
A = legend_algebraic.copy()
C = legend_constants.copy()
S = legend_states.copy()

df_protocol = config['experimental_conditions']['individual']['protocol']
df_protocol = df_protocol.drop('t', axis = 1)
df_initial_state_protocol = config['experimental_conditions']['individual']['initial_state_protocol']


filename_so = config['filename_so'] 
Ina = InaModel(filename_so)

data_no_grad = data.copy()

weight = config['experimental_conditions']['individual']['sample_weight']
weight_grad = config['experimental_conditions']['individual']['sample_derivative_weight']

m_index = config['runtime']['m_index']
params_list = list(config['runtime']['m_index'].get_level_values(1))
const_from_sol_best = legend_constants.copy()
mask_mult = config['runtime']['mask_multipliers']

ind_alpha = []
delete_mat_ind = []
m_ind_small = config['runtime']['m_index'].copy()
ind_small = []

ind_numeric=[]
#Parameters that we do not want to sample
pass_params = [     
             #'c_m', 'R',
#                 'g_max',
               #'x_c_comp', 'x_r_comp',
                'alpha'
#              'a0_m', 'b0_m', 's_m', 'delta_m', 
#               'tau_m_const',
#             'a0_h', 'b0_h', 's_h', 'delta_h', 
#               'tau_h_const',
              #'a0_j', 'b0_j', 's_j', 'delta_j', 'tau_j_const',
#                 'v_half_m', 'k_m', 
#                 'v_half_h', 'k_h', 
#                 'v_rev', 'g_leak', 
#             'tau_z'
              ]



for param in pass_params:
    ind = 'common'
    if param in config['experimental_conditions']['individual']['params']:
        ind = 'individual'
    ind_numeric.append(np.where(m_index==(ind, param)))
    delete_mat_ind.append(np.where(m_ind_small==(ind, param)))
    ind_alpha.append((ind, param))

m_index = m_index.delete(ind_numeric)
#print(f'ind_alpha={ind_alpha}\n')

bounds = np.delete(bounds, ind_numeric, 0)
sol_best_before = np.delete(sol_best.values, ind_numeric)


if 'v_half_h' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]] = np.array([30., 100.])
if 'k_h' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'k_h')[0][0]] = np.array([3., 20.])
if 'v_half_m' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'v_half_m')[0][0]] = np.array([-10., 60.])
if 'k_m' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'k_m')[0][0]] = np.array([3., 20.])
if 'g_leak' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'g_leak')[0][0]] = np.array([0.01, 20.])
if 'g_max' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'g_max')[0][0]] = np.array([0.005, 20.])
if 'v_rev' not in pass_params:
    bounds[np.where(m_index.get_level_values(1) == 'v_rev')[0][0]] = np.array([1., 150.])

if GRAD:
    flag = 'grad'
    data = np.gradient(data)
else:
    flag = 'ina'


if LOG :
    mask_log = np.delete(np.array(mask_mult), ind_numeric, 0)
    for i, mult in enumerate(mask_log):
        if mult:
            bounds[i] = np.log10(bounds[i])
            sol_best_before[i] = np.log10(sol_best_before[i])
    matrix_name = 'static_cov_mat.npy'

else:
    mask_log = None    
    matrix_name = 'static_cov_mat.npy'
matrix_name = 'proposal_fixed.npy' #proposal matrix
param_cov_mat = np.load(matrix_name)

param_cov_mat = np.delete(np.delete(param_cov_mat, delete_mat_ind, axis=0), delete_mat_ind, axis=1)
param_cov_mat = param_cov_mat*len(sol_best_before)/28

#Some parameters are multipliers. The routine below converts them to absolute values.
const_from_sol_best = legend_constants.copy()
for i, ind in zip(ind_numeric, ind_alpha):
    if mask_mult[i[0][0]]:
        const_from_sol_best[ind[1]] *= sol_best[ind]
    else:
        const_from_sol_best[ind[1]] = sol_best[ind]

trace = func_model(sol_best_before,
                    m_index,
                    Ina=Ina,
                    const=const_from_sol_best,
                    config=config,
                    flag = flag,
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
                    flag = flag,
                    mask_log = mask_log,
                    ) 

downsampl = 5  #downsampling
n_steps=df_protocol.shape[1] #number of voltage step sweeps
len_step = int(len(data)/n_steps)
weight_cut = weight.drop('t', axis = 1)

mask_cut = np.zeros_like(weight_cut)
mask_cut_big = mask_cut.copy()
mask_cut_big[(np.where(weight_cut > 50.)[0][::1], 
              np.where(weight_cut > 50.)[1][::1])]= 1.
mask_cut_big = mask_cut_big.reshape(-1)
mask_cut_big = mask_cut_big.astype('bool')
mask_cut[(np.where(weight_cut > 1.)[0][::downsampl], 
              np.where(weight_cut > 1.)[1][::downsampl])]= 1.

mask_cut_down = mask_cut.reshape(-1)
mask_cut_down = mask_cut_down.astype('bool')

data_cut_size = np.sum(mask_cut_down)
data_cut = np.zeros_like(data_cut_size)
delta_data = np.array(data - trace)
data_cut = delta_data[mask_cut_down.astype(bool)]

n = 1
p = np.shape(data_cut)[0]


S_0 = np.eye(p) * data_cut**2
        

diff_no_ina = np.sum((data[mask_cut_down] - no_ina[mask_cut_down])**2)
diff_best = np.sum((data[mask_cut_down] - trace[mask_cut_down])**2)

ina_dict = {}
ina_dict['data'] = data
ina_dict['m_index'] = m_index
ina_dict['Ina'] = Ina
ina_dict['const'] = const_from_sol_best
ina_dict['config'] = config
#ina_dict['shape'] = shape
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
            if s_b_b[i]<0:
                ub, lb = s_b_b[i]*up, s_b_b[i]*low
                if lb < param_bounds[0]:
                    lb = param_bounds[0]*0.99
                if ub > param_bounds[1]:
                    ub = param_bounds[1]*1.01

            else:
                lb, ub = s_b_b[i]*up, s_b_b[i]*low
                if lb < param_bounds[0]:
                    lb = param_bounds[0]*1.01
                if ub > param_bounds[1]:
                    ub = param_bounds[1]*0.99

            new_bounds.append([lb, ub])
        new_bounds = np.array(new_bounds)
        new_bounds[np.where(m_index.get_level_values(1) == 'v_half_m')[0][0]] = np.array([20.0, 30.])
        new_bounds[np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]] = np.array([60., 90.])

        new_bounds[np.where(m_index.get_level_values(1) == 'k_m')[0][0]] = np.array([8., 12.])
        new_bounds[np.where(m_index.get_level_values(1) == 'k_h')[0][0]] = np.array([5., 15.])
#Initial states of chain are generated below within new_bounds.
#For some reason we use pymcmcstat here. This is a leftover from previous version, but it works. 
    initial_value = pymcmcstat.ParallelMCMC.generate_initial_values(1,
                                                                    len(new_bounds),
                                                                    new_bounds.T[0],
                                                                    new_bounds.T[1])
    
    
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
               flag = flag,
               mask_log = mask_log)
    check_time += time.time()
    diff_v = initial_value[0][np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]] - initial_value[0][np.where(m_index.get_level_values(1) == 'v_half_m')[0][0]] 


    diff_trace_now = np.sum((data[mask_cut_down] - trace_now[mask_cut_down])**2)
    if check_time < 2.0 and diff_v > 30 and diff_trace_now < diff_no_ina:
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
print(f'start_dicts = {start_dicts}')

model = pm.Model()
transform = None
scale = 1   #Proposal matrix is empirical covariance matrix multiplied by scale factor.

tune_interval = 100  #Proposal matrix is adaptive change it every N steps. 10_000 if initial proposal is good enough.
mat = matrix_name.split('.')[0]
scale_factor = 1/5  #scaling factor used for delayed rejection

FOLDER_OUT = '../results/pymc/inactivation_for_paper'#count_three_starts'

fold = f'old_downsample_{GRAD}_grad_{LOG}_log_{PIPETTE}_pipette_{nchain}_chain_{ndraws}_draws_{tune_interval}_tune_{mat}_matrix_{scale_factor}_scale_factor' 

print(f'sol_best_before = {sol_best_before}')

mu=(bounds.T[0]+bounds.T[1])/2
sigma=(bounds.T[1]-bounds.T[0])*2.
mu[np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]]=76
sigma[np.where(m_index.get_level_values(1) == 'v_half_h')[0][0]]=9
mu[np.where(m_index.get_level_values(1) == 'k_h')[0][0]]=11
sigma[np.where(m_index.get_level_values(1) == 'k_h')[0][0]]=3
mu[np.where(m_index.get_level_values(1) == 'v_rev')[0][0]]=18
sigma[np.where(m_index.get_level_values(1) == 'v_rev')[0][0]]=7

bounds.T[0][mask_log]-=2
bounds.T[1][mask_log]+=2

with model:
    parameters = pm.Normal('parameters',
                        mu=  mu,
                        sigma=sigma,
                        transform = transform)


    loglike = CustomLogLike(ina_dict=ina_dict,
                            n=n,
                            p=p,
                            S=S_0,
                            )

    params = as_tensor_variable(parameters,)
    logl_mu = loglike(params)
    pm.Potential("likelihood", logl_mu)

    step_parameters = DRAM([parameters],
                             loglike = loglike, 
                             S = param_cov_mat, 
                             proposal_dist=pm.MultivariateNormalProposal,
                             scaling=scale,
                             tune = 'S',
                             tune_interval = tune_interval,
                             bounds = bounds.T, 
                             initial_values_size = len(sol_best_before),
                             transform = transform,
                             scale_factor = scale_factor,
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

foldername = os.path.join(FOLDER_OUT, fold, trace_name)

fold_list = foldername.split('/')
for k in range(1, len(fold_list)+1):
    loc_path = os.path.join(*foldername.split('/')[:k])
    logic = os.path.isdir(loc_path)
    if not logic:
        os.mkdir(loc_path)

time_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
filename_last = os.path.join(foldername, time_suffix+'.nc')
idata.to_netcdf(filename_last)
print("ALL DONE")

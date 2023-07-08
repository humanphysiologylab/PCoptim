from distutils.dep_util import newer_group
from re import M
import numpy as np
import pandas as pd
import sys
sys.path.append('../../pypoptim/mpi_scripts/')
from gene_utils import update_C_from_genes


def func_model(Params, 
               m_index,
               Ina=None,
               const=None,
               config=None,
               flag="ina",
               mask_log=None,
               ):

    params = Params.copy()
    if flag not in {"ina", "grad"}:
        raise ValueError('The parameter "flag" must be one of {ina, grad}')
    
    if mask_log is not None:
#        print(f'params={params}')
        params[mask_log] = 10**params[mask_log]

    C = const.copy()
    A = config['runtime']['legend']['algebraic'].copy()
    S = config['runtime']['legend']['states'].copy()
    
    df_protocol = config['experimental_conditions']['individual']['protocol'] 
    df_initial_state_protocol = config['experimental_conditions']['individual']['initial_state_protocol']

    genes = pd.Series(params, index=m_index)
    update_C_from_genes(C, genes, 'individual', config)
    full_data = Ina.run(A,S,C,df_protocol, df_initial_state_protocol)

    if flag == 'ina':    
        result = np.array(full_data.I_out)
    if flag == 'grad':
        result = np.array(np.gradient(full_data.I_out))
    return result


def return_data_cut(Params,
                    data=None, 
                    m_index=None, 
                    Ina=None, 
                    const=None, 
                    config=None,
#                    shape=[19, 5000],
                    mask_log=None,
                    mask_cut=None,
                    flag='ina',
                    ):
#    n_step, len_step = shape
    
    if mask_cut is None:
        mask_cut = np.ones(len_step).astype('bool')
    
    ina = func_model(Params, 
                    m_index, 
                    Ina=Ina, 
                    const=const, 
                    config=config, 
                    flag=flag, 
                    mask_log=mask_log)  
    if np.any(np.isnan(ina)):
        return np.float64(-np.inf)

    delta_ina = data - ina    
    data_cut_size = int(np.sum(mask_cut))

    data_cut = np.zeros(data_cut_size)
    data_cut = delta_ina[mask_cut]

    return data_cut

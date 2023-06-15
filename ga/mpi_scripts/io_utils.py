import os
import json
# import git
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

from gene_utils import create_genes_dict_from_config, \
                       create_constants_dict_from_config, \
                       generate_bounds_gammas_mask_multipliers


def prepare_config(config_filename):

    config_path = os.path.dirname(os.path.realpath(config_filename))

    with open(config_filename) as f:
        text = f.read()
        config = json.loads(text)

    config['runtime'] = dict()

    config['runtime']['config_path'] = config_path
    config['runtime']['filename_so_abs'] = os.path.normpath(os.path.join(config_path, config['filename_so']))
    config['runtime']['genes_dict'] = create_genes_dict_from_config(config)
    config['runtime']['constants_dict'] = create_constants_dict_from_config(config)

    m_index_tuples = [(exp_cond_name, gene_name) for exp_cond_name, gene in config['runtime']['genes_dict'].items() for
                      gene_name in gene]
    m_index = pd.MultiIndex.from_tuples(m_index_tuples)
    m_index.names = ['ec_name', 'g_name']

    config['runtime']['m_index'] = m_index

    legend = dict()
    legend['states'] = pd.read_csv(os.path.normpath(os.path.join(config_path, config["filename_legend_states"])),
                                   usecols=['name', 'value'], index_col='name')['value']  # Series
    legend['constants'] = pd.read_csv(os.path.normpath(os.path.join(config_path, config["filename_legend_constants"])),
                                      usecols=['name', 'value'], index_col='name')['value']  # Series
    legend['algebraic'] = pd.read_csv(os.path.normpath(os.path.join(config_path, config["filename_legend_algebraic"])),
                                      usecols=['name', 'value'], index_col='name')['value']  # Series
    config['runtime']['legend'] = legend

    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue

        filename_phenotype = os.path.normpath(os.path.join(config_path, exp_cond['filename_phenotype']))
        phenotype = pd.read_csv(filename_phenotype)
        phenotype_dt = phenotype['t'][1] - phenotype['t'][0]
        exp_cond['phenotype'] = phenotype.drop('t', axis=1).values.reshape(-1)
        exp_cond['filename_phenotype'] = filename_phenotype
        
        protocol = pd.read_csv(os.path.normpath(os.path.join(config_path, 
                                                             exp_cond["filename_protocol"])))
        protocol_dt = protocol['t'][1] - protocol['t'][0]
        exp_cond['protocol'] = protocol
        
        
        if 'filename_initial_state_protocol' in exp_cond:
            initial_state_protocol = pd.read_csv(os.path.normpath(os.path.join(config_path, 
                                                                           exp_cond["filename_initial_state_protocol"])))
        elif 'initial_state_protocol' in exp_cond:
            if 'v' in exp_cond['initial_state_protocol'] and 't' in exp_cond['initial_state_protocol']:
                initial_state_protocol = pd.DataFrame()
                initial_state_protocol['t'] = np.arange(0, exp_cond['initial_state_protocol']['t'], protocol_dt)
                initial_state_protocol['v'] = exp_cond['initial_state_protocol']['v']           
            else:
                raise ValueError("'Initial_state_protocol' should contain 'v' and 't' in the same units as protocol.\nExample:'initial_state_protocol':{'v':-80., 't':0.6} ")

        else: 
            raise ValueError("Config should contain 'filename_initial_state_protocol' or 'initial_state_protocol'")
        exp_cond['initial_state_protocol'] = initial_state_protocol
        # Tests 
        
        if phenotype.shape != protocol.shape:
            raise ValueError(f"Protocol and phenotype should have same shape, now protocol shape is {protocol.shape}, phenotype shape is {phenotype.shape}")
        
        if 't' not in protocol.columns:
            raise ValueError(f"Column 't' in protocol not found")
        
        if 't' not in phenotype.columns:
            raise ValueError(f"Column 't' in phenotype not found")
        
        if phenotype_dt != protocol_dt:
            raise ValueError(f"Protocol and phenotype have different time step: protocol time step is {protocol_dt} s., phenotype time step is {phenotype_dt} s.")

        
        if 'filename_sample_weight' in exp_cond:
            filename_sample_weight = os.path.normpath(os.path.join(config_path, exp_cond['filename_sample_weight']))
            sample_weight = pd.read_csv(filename_sample_weight)
            exp_cond['sample_weight'] = sample_weight.w
            if 'w_grad' in sample_weight.columns:
                exp_cond['sample_weight_grad'] = sample_weight.w_grad
            exp_cond['filename_sample_weight'] = filename_sample_weight

    bounds, gammas, mask_multipliers = generate_bounds_gammas_mask_multipliers(config['runtime']['genes_dict'])
    config['runtime']['bounds'] = bounds
    config['runtime']['gammas'] = gammas
    config['runtime']['mask_multipliers'] = mask_multipliers

    config['runtime']['kw_ga'] = dict(crossover_rate=config.get('crossover_rate', 1.0),
                                      mutation_rate=config.get('mutation_rate', 0.1),
                                      gamma=config.get('gamma', 1.0))
    seed = config.get('seed', None)
    if seed is None:
        sq = np.random.SeedSequence()
        seed = sq.entropy
    config['runtime']['seed'] = seed

    return config


def update_output_dict(config):

    folder = os.path.normpath(os.path.join(config['runtime']['config_path'],
                                                       config.get("output_folder_name", "./results")))
    time_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    folder = os.path.join(folder, time_suffix)
    config['runtime']['time_suffix'] = time_suffix

    config['runtime']['output'] = dict(folder=folder,
                                       folder_dump=os.path.join(folder, "dump"),
                                       folder_best=os.path.join(folder, "best"),
                                       folder_phenotype=os.path.join(folder, "phenotype"),
                                       )


def backup_config(config):
    filename = os.path.join(config['runtime']['output']['folder'], "config_backup.pickle")
    with open(filename, "wb") as f:
        pickle.dump(config, f)
    config['runtime']['output']['config_backup'] = filename


def dump_dict(dct, folder):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for key, value in dct.items():

        filename = os.path.join(folder, key)
        if not os.path.isfile(filename):
            with open(filename, "wb") as _:
                pass

        with open(filename, 'ba+') as f:
            np.asarray(value).tofile(f)


def dump_epoch(recvbuf_dict, config):
    dump_dict(recvbuf_dict, config['runtime']['output']['folder_dump'])


def save_sol_best(sol_best, config):

    output_dict = config['runtime']['output']

    genes = pd.Series(sol_best.x, index=config['runtime']['m_index'])
    filename = os.path.join(output_dict['folder'], 'sol_best.csv')
    genes.to_csv(filename)

    for exp_cond_name in config['experimental_conditions']:
        if exp_cond_name == 'common':
            continue

        folder_phenotype = config['runtime']['output']['folder_phenotype']
        if not os.path.isdir(folder_phenotype):
            os.mkdir(folder_phenotype)

        df = sol_best['phenotype'][exp_cond_name]

        # Rewrite last epoch
        filename = os.path.join(folder_phenotype, f"phenotype_{exp_cond_name}.csv")
        df.to_csv(filename, index=False)

        # Append last epoch to previous
        filename = os.path.join(folder_phenotype, f"phenotype_{exp_cond_name}")
        if not os.path.isfile(filename):
            with open(filename, "wb") as f:
                pass

        with open(filename, 'ba+') as f:
            df.values.astype(np.float32).tofile(f)

    d = dict(genes=sol_best.x,
             loss=sol_best.y,
             status=sol_best.status)

    folder_best = output_dict['folder_best']
    dump_dict(d, folder_best)


def collect_results(case, dirname_results, dump_keys=None):

    config_path = os.path.join(dirname_results, case)
    with open(os.path.join(config_path, "config_backup.pickle"), 'rb') as f:
        config = pickle.load(f)

    if dump_keys is None:
        dump_keys = ['dump', 'best']

    dump = {}
    for folder in dump_keys:
        dump[folder] = {}
        for key in 'genes', 'status', 'loss':
            filename = os.path.join(config_path, folder, key)
            if os.path.isfile(filename):
                dump[folder][key] = np.fromfile(filename)

    filename = os.path.join(config_path, 'sol_best.csv')
    if os.path.isfile(filename):
        sol_best = pd.read_csv(filename, index_col=[0, 1]).iloc[:, -1]
    else:
        sol_best = None

    phenotype_best = {}

    for exp_cond_name in config['experimental_conditions']:

        if exp_cond_name == 'common':
            continue

        filename = os.path.join(config_path, "phenotype", f"phenotype_{exp_cond_name}.csv")
        if os.path.isfile(filename):
            try:
                phenotype_best[exp_cond_name] = pd.read_csv(filename)
            except pd.errors.EmptyDataError as e:
                print(f'{filename} is empty')
                continue

    results = dict(config=config,
                   dump=dump,
                   sol_best=sol_best,
                   phenotype_best=phenotype_best)

    return results

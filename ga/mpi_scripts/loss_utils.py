from pypoptim.losses import RMSE
import numpy as np
import logging
logger = logging.getLogger(__name__)


def calculate_loss(sol, config):

    loss = 0
    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue
        
        if config['loss'] == 'RMSE':
            x = sol['phenotype'][exp_cond_name]['I_out']
            y = exp_cond['phenotype']['I_out']
            sample_weight = exp_cond.get('sample_weight', None)
            loss += RMSE(x, y, sample_weight=sample_weight)
        
        elif config['loss'] == 'RMSE_GRAD':
            
            x = sol['phenotype'][exp_cond_name]['I_out']
            x_grad = sol['phenotype'][exp_cond_name]['grad']
            
            y = exp_cond['phenotype']['I_out']
            y_grad = np.gradient(y)
            
            sample_weight = exp_cond.get('sample_weight', None)
            sample_weight_grad = exp_cond.get('sample_weight_grad', None)
            
            loss += RMSE(x, 
                    y,
                    squared=True,
                    sample_weight=sample_weight) * 0.025**2
            
            loss += RMSE(x_grad, 
                    y_grad, 
                    squared=True,
                    sample_weight=sample_weight_grad) * 0.975**2
       
    logger.info(f'loss = {loss}')

    return loss

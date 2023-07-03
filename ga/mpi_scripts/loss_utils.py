from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import logging
logger = logging.getLogger(__name__)


def RMSE(x, y, *,squared=False, sample_weight=None, multioutput="uniform_average"):
    return MSE(
        x, y, squared=squared, sample_weight=sample_weight, multioutput=multioutput
    )
def calculate_loss(sol, config):

    loss = 0
    model_column = config.get("column_model", "I_out")


    for exp_cond_name, exp_cond in config['experimental_conditions'].items():


        if exp_cond_name == 'common':
            continue
        
        sample_weight = exp_cond.get('sample_weight', None)
        phenotype = exp_cond['phenotype']
        model_output = sol['phenotype'][exp_cond_name][model_column].values.reshape(-1)
            
        if config['loss'] == 'RMSE':
            loss += RMSE(model_output, phenotype, sample_weight=sample_weight)

        elif config['loss'] == 'RMSE_GRAD':
            
            d_phenotype = np.gradient(phenotype)
            d_model_output = np.gradient(model_output)

            sample_derivative_weight = exp_cond.get('sample_derivative_weight', None)
            
            loss += RMSE(phenotype, 
                    model_output,
                    squared=True,
                    sample_weight=sample_weight) * 0.025**2
            
            loss += RMSE(d_phenotype, 
                    d_model_output, 
                    squared=True,
                    sample_weight=sample_derivative_weight) * 0.975**2
       
    logger.info(f'loss = {loss}')

    return loss

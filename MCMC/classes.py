import scipy.stats as ss
import pytensor.tensor as at
import os
import numpy as np
import pymc as pm
from fastprogress import fastprogress
fastprogress.printing = lambda: True
from pymc.step_methods.arraystep import ArrayStepShared, metrop_select
from pymc.step_methods.compound import Competence
from pymc.step_methods.metropolis import MultivariateNormalProposal, delta_logp
from pymc.blocking import DictToArrayBijection, RaveledVars
from typing import  Any, List, Tuple, Dict
from pymc.pytensorf import floatX
from pymc.step_methods.metropolis import tune
import logging 

logger = logging.getLogger()
# logger.disabled = True
logger.setLevel(logging.CRITICAL)
# logger.setLevel(logging.DEBUG)


class CustomLogLike(at.Op):
    
    itypes = [at.dvector]
    otypes = [at.dscalar] 

    def __init__(self, 
                 ina_dict=None, 
                 loglike=None, 
                 n=None, 
                 p=None, 
                 S=None,
                ):

#        print('Updating beta, phi, cov_mat from inverse wishart distribution')
        
        self.data = ina_dict['data']
        self.m_index = ina_dict['m_index']
        self.Ina = ina_dict['Ina']
        self.const = ina_dict['const']
        self.config = ina_dict['config']

        self.mask_log = ina_dict['mask_log']
        self.mask_cut = ina_dict['mask_cut']

        self.mask_cut_big = ina_dict['mask_cut_big'] if 'mask_cut_big' in ina_dict.keys() else ina_dict['mask_cut'].copy()     
        self.return_data_cut = ina_dict['return_data_cut']


        self.flag = ina_dict['flag']

        self.likelihood = loglike
        self.n = n
        self.p = p

        self.S = S
        self.G = 1


        
    def perform(self, node, inputs, outputs,):
        (params,) = inputs 

        data_cut = self.return_data_cut(params,
                                data=self.data,
                                m_index=self.m_index,
                                Ina=self.Ina,
                                const=self.const,
                                config=self.config,
                                mask_log=self.mask_log,
                                mask_cut=self.mask_cut_big,
                                flag=self.flag,
                                )

        logging.debug(f'CustomLoglike: perform step, params = {params}, type = {type(params)}')
        if type(data_cut) != np.ndarray:
            # logging.debug(f'None in data_cut')
            logl = np.array(data_cut)
        else:
            logl = -np.sum(data_cut**2)/self.G

        logging.debug(f'Custom loglike: perform step: loglike = {logl}')
        outputs[0][0] = np.array(logl)
    def update_cov_matrix(self, params_accepted=None):
        logging.debug(f'CustomLoglike: perform step, params_accepted = {params_accepted}, type = {type(params_accepted)}')

        data_cut = self.return_data_cut(params_accepted,
                                data=self.data, 
                                m_index=self.m_index, 
                                Ina=self.Ina, 
                                const=self.const, 
                                config=self.config,
                                mask_log=self.mask_log,
                                mask_cut=self.mask_cut_big,
                                flag=self.flag,

                                )
        if type(data_cut) != np.ndarray:
            logging.debug(f'None in data_cut')
        else:
            self.G = ss.invgamma.rvs(len(data_cut)/2,
                                    scale = np.sum((data_cut)**2)/2)


class DRAM(ArrayStepShared):
    name = "DRAM"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": bool,
            "tune": bool,
            "scaling": np.float64,
            "dram_level": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        loglike = None,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,      #proposal matrix scaling
        tune='S',
        tune_interval=100,  #update proposal matrix every tune_interval steps
        tune_drop_fraction: float = 0.9,
        model=None,
        bounds=None,
        initial_values_size=None,
        transform=True,
        scale_factor=1/5,   #Delayed rejection  scaling
        foldername=None,    #Folder to save parameters covariance matrix
        S_weight=1000,      #Proposal distribution prior. Set ~ to the number of samples used to estimate parameters covariance.
        **kwargs
    ):
        model = pm.modelcontext(model)
        initial_values = model.initial_point()
        self.loglike = loglike

        if initial_values_size is None:
            initial_values_size = sum(initial_values[n.name].size 
            for n in model.value_vars)

        if vars is None:
            vars = model.cont_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(initial_values_size)

        self.S = S.copy()
        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(self.S)
        else:
            self.proposal_dist = UniformProposal(self.S)
        
        self.scaling = np.atleast_1d(scaling).astype("d")
        if tune not in {None, "scaling", "S"}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, S}')
        self.tune = True
        self.tune_target = tune
        self.tune_interval = tune_interval
        self.tune_drop_fraction = tune_drop_fraction
        self.steps_until_tune = tune_interval
        self.accepted = False
        self.dram_level = 0.
        self.S_weight = S_weight

        self._history = []
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling,
            steps_until_tune=tune_interval,
            accepted=self.accepted,
            dram_level=self.dram_level,
        )

        self.bounds = bounds
        self.initial_values_size = initial_values_size
        self.transform = transform
        self.scale_factor = scale_factor
        self.tune_over = False
        if foldername is not None:
            self.foldername = f'../data/matrix/{foldername}'
            fold_list = self.foldername.split('/')
            for k in range(2, len(fold_list)+1):
                loc_path = os.path.join(*self.foldername.split('/')[:k])
                logic = os.path.isdir(loc_path)
                if not logic:
                    os.mkdir(loc_path)
            print(f'Saving covariance matrix in {os.path.abspath(self.foldername)}')
        else:
            self.foldername = None
        shared = pm.make_shared_replacements(initial_values, vars, model)
        self.delta_logp = delta_logp(initial_values, model.logp(), vars, shared)
        super().__init__(vars, shared)


    def reset_tuning(self):
        """Resets the tuned sampler parameters and history to their initial values."""
        # history can't be reset via the _untuned_settings dict because it's a list
        self._history = []
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:
        point_map_info = q0.point_map_info
        q0 = q0.data
        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune and not self.tune_over: #and self.tune:
            
            if self.tune_target == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
               
            elif self.tune_target == "S":
                history_weight = len(self._history)
                magic_coeff = 2.38**2/self.initial_values_size
                self.S = magic_coeff * (history_weight * np.cov(self._history,rowvar=False) 
                        + self.S_weight * self.S) / (history_weight + self.S_weight)
                self.proposal_dist = pm.MultivariateNormalProposal(self.S)

                
            # Reset counter
            self.steps_until_tune = self.tune_interval
            # self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        if not self.transform:
            while(np.any(q0 + epsilon <= self.bounds[0]) or np.any(q0 + epsilon >= self.bounds[1])): 
                epsilon = self.proposal_dist() * self.scaling

        it = len(self._history)

        # propose a jump
        q = floatX(q0 + epsilon)
            
        dellog_q0_q = self.delta_logp(q, q0)
        logging.debug(f'dellog_q0_q = {dellog_q0_q}')

        q_new, accepted = metrop_select(dellog_q0_q, q, q0)
        accept = dellog_q0_q
        self.dram_level = 0.

        if not accepted: #Delayed Rejection
            self.dram_level = 1.
            epsilon = self.proposal_dist() * self.scaling * self.scale_factor
            q2 = floatX(q0 + epsilon)
            if not self.transform:
                while(np.any(q2 <= self.bounds[0]) or np.any(q2 >= self.bounds[1])): 
                    epsilon = self.proposal_dist() * self.scaling * self.scale_factor
                    q2 = floatX(q0 + epsilon)
                    
            dellog_q0_q2 = self.delta_logp(q2, q0)
            dellog_q2_q = dellog_q0_q - dellog_q0_q2
            logging.debug(f'dellog_q0_q2 = {dellog_q0_q2}')
            logging.debug(f'dellog_q2_q = {dellog_q2_q}')


            if dellog_q2_q > 0: 
                accept = -np.inf
            else:

                alpha_ratio = np.log(1 - np.exp(dellog_q2_q)) - np.log(1 - np.exp(dellog_q0_q))
                scaled_S_inv = np.linalg.inv(self.scaling**2*self.S) #accounted for scaling
                prop_ratio = -(np.dot(q2 - q, np.dot(scaled_S_inv, q2 - q)) 
                            - np.dot(q0 - q, np.dot(scaled_S_inv, q0 - q)))/2
                accept = dellog_q0_q2 + prop_ratio + alpha_ratio 
                
            q_new, accepted = metrop_select(accept, q2, q0)

        logging.debug(f'Level {self.dram_level} Delayed Rejection')
        logging.debug(f'Accept = {accept}, Accepted = {accepted}')


        self.accepted += accepted
        self._history.append(q_new)

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "accept": np.exp(accept),
            "accepted": accepted,
            "dram_level": self.dram_level,
        }

        self.loglike.update_cov_matrix(params_accepted=q_new)
        q_new = RaveledVars(q_new, point_map_info)

        return q_new, [stats]


    def stop_tuning(self):
        """At the end of the tuning phase, this method removes the first x% of the history
        so future proposals are not informed by unconverged tuning iterations.
        """
        it = len(self._history)
        n_drop = int(self.tune_drop_fraction * it)
        self._history = self._history[n_drop:]
        return super().stop_tuning()


    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE

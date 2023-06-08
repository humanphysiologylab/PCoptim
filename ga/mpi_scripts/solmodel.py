import sys
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from pypoptim.algorythm import Solution

from gene_utils import update_C_from_genes

from loss_utils import calculate_loss

sys.path.append('src/python_func/')
from models import calculate_tau_j_from_ab,calculate_tau_h_from_ab,calculate_tau_m_from_ab 


class SolModel(Solution):

    def __init__(self, x, **kwargs_data):
        super().__init__(x, **kwargs_data)
        for attr in 'model', 'config':
            if not hasattr(self, attr):
                raise AttributeError(attr, "make this guy static")

        self._status = None
        self.__status_valid = 2

    @property
    def status(self):
        return self._status

    def update(self):

        self['phenotype'] = {}

        legend = self.config['runtime']['legend']

        genes = pd.Series(self.x, index=self.config['runtime']['m_index'])

        for exp_cond_name, exp_cond in self.config['experimental_conditions'].items():

            if exp_cond_name == 'common':
                continue

            C = legend['constants'].copy()
            S = legend['states'].copy() 
            A = legend['algebraic'].copy()

            update_C_from_genes(C, genes, exp_cond_name, self.config)
            #check that tau_h, tau_m and tau_j not so fast
            tau_m_right = calculate_tau_m_from_ab(15., C)
            tau_h_right = calculate_tau_h_from_ab(15., C)
            tau_j_right = calculate_tau_j_from_ab(15., C)
            if (tau_m_right < 5e-5) or (tau_h_right < 7e-5) or (tau_j_right < 2e-4):
                self._x = genes.values
                self._y = np.nan
                return

            df_protocol = exp_cond['protocol']
            df_initial_state_protocol = exp_cond['initial_state_protocol']

            pred = self.model.run(A,
                                  S,
                                  C,
                                  df_protocol,
                                  df_initial_state_protocol,
                                  )
            self._status = self.model.status
            if (self._status != self.__status_valid) or (np.any(np.isnan(pred))):

                self._x = genes.values
                self._y = np.nan
                return
            self['phenotype'][exp_cond_name] = pred.copy()

        assert np.all(self._x == genes)
        self._y = calculate_loss(self, self.config)

    def is_all_equal(self, other, keys_check=None):
        if not np.allclose(self.x, other.x):
            x = np.vstack([self.x, other.x, self.x - other.x]).T
            logger.info(f"`x`s differs: {x}")
            return False

        if self.y != other.y:
            logger.info(f"`y`s differs: {self.y} {other.y}")
            return False

        if keys_check is None:
            keys_check = ['state']
        for key in keys_check:
            if key in self:
                if key in other:
                    if not np.allclose(self[key], other[key]):
                        logger.info(f"`{key}`s differs")
                        return False
                else:
                    logger.info(f"`{key}` is not in `other`")
                    return False
            else:
                if key in other:
                    logger.info(f"`{key}` is not in `self`")
                    return False

        return True

    def is_valid(self):
        if not self.is_updated():
            return False
        else:
            flag_valid = self._status == self.__status_valid and np.isfinite(self._y)
            if 'phenotype' not in self:  # solution was gathered via MPI
                return flag_valid
            else:
                return flag_valid and all(not np.any(np.isnan(p)) for p in self['phenotype'].values())

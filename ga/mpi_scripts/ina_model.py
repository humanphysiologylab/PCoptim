from multiprocessing.reduction import recv_handle
import os
import ctypes
import pandas as pd
import numpy as np


class InaModel:

    def __init__(self, filename_so):

        filename_so_abs = os.path.abspath(filename_so)
        ctypes_obj = ctypes.CDLL(filename_so_abs)
        ctypes_obj.run.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            ctypes.c_int,
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'
            )
        ]
        ctypes_obj.run.restype = ctypes.c_int

        self._run = ctypes_obj.run
        self._status = None

    @property
    def status(self):
        return self._status

    def run(self, A, S, C, df_protocol, df_initial_state_protocol,
            return_algebraic=False, **kwargs):

        t0 = df_initial_state_protocol.t.values
        v0 = df_initial_state_protocol.v.values
        initial_state_len = len(t0)

        initial_state_S = np.zeros((initial_state_len, len(S)))
        initial_state_A = np.zeros((initial_state_len, len(A)))

        self._run(
            S.values.copy(),
            C.values.copy(),
            t0,
            v0,
            initial_state_len,
            initial_state_S,
            initial_state_A
            )

        t = df_protocol.t.values
        v_all = df_protocol.drop('t', axis=1)
        len_one_sweep = len(t)
        output_len = v_all.size

        S_output = np.zeros((output_len, len(S)))
        A_output = np.zeros((output_len, len(A)))
        
        S0 = initial_state_S[-1].copy()

        for sweep in v_all:
            n_sweep = int(sweep)
            start, end = len_one_sweep*n_sweep, len_one_sweep*(n_sweep+1) 
            v = np.array(v_all[sweep])
            self._status = self._run(
                S0,
                C.values.copy(),
                t,
                v,
                len_one_sweep,
                S_output[start:end], 
                A_output[start:end],
                )
        df_A = pd.DataFrame(A_output, columns=A.index)
        df_S = pd.DataFrame(S_output, columns=S.index)
        df_S['grad'] = np.gradient(df_S.I_out)
        
        if return_algebraic:
            return df_S, df_A
        return df_S

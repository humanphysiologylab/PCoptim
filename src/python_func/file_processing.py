import numpy as np
import pandas as pd
from pyabf import ABF, waveform


def trace_from_atf(path, save=False, new_path=None,):
    "Default path for new traces is '../data/traces/'"
    
    raw_data = pd.read_csv(path,
                       skiprows=10,
                       delimiter='\t',
                       index_col=0
                      )
    dt = raw_data.index[1]
    t = np.arange(raw_data.shape[0]*raw_data.shape[1])*dt
    trace = pd.DataFrame(raw_data.values.reshape(-1), index=t, columns=['I_out'])
    trace.index.name = 't'

    if save:
        if not new_path:
            new_path = path.split('/')[-1].split('.atf')[0]
            new_path = f'../data/traces/{new_path}.csv'
        trace.to_csv(new_path)
        print(f'Trace saved in {new_path}')
    return trace


def trace_from_abf(path, save=False, new_path=None,):
    "Default path for new traces is '../data/traces/'"

    raw_data = ABF(path)
    data = raw_data.data[0]
    dt = raw_data.dataSecPerPoint
    t = np.arange(0, len(data)*dt, dt)
    trace = pd.DataFrame(data, index=t, columns=['I_out'])
    trace.index.name = 't'
    if save:
        if not new_path:
            new_path = path.split('/')[-1].split('.abf')[0]
            new_path = f'../data/traces/{new_path}.csv'
        trace.to_csv(new_path)
        print(f'Trace saved in {new_path} file')
    return trace

def protocol_from_abf(path, save=False, new_path=None,):
    "Default path for new traces is '../data/protocols/'"

    raw_data = ABF(path)
    dt = raw_data.dataSecPerPoint
    t = np.arange(0, raw_data.sweepPointCount*dt, dt)
    epochTable = waveform.EpochTable(raw_data, 0)
    df_protocol = pd.DataFrame(t, columns=['t']).set_index('t')
    for i, sweep in enumerate(epochTable.epochWaveformsBySweep):
        df_protocol[i] = sweep.getWaveform()

    if save:
        if not new_path:
            new_path = path.split('/')[-1].split('.abf')[0]
            new_path = f'../data/protocols/{new_path}.csv'
        df_protocol.to_csv(new_path)
        print(f'Protocol saved in {new_path} file')
    return df_protocol

def initial_protocol_from_abf(path, save=False, new_path=None,):
    "Default path for new traces is '../data/initial_protocols/'"

    raw_data = ABF(path)
    dt = raw_data.dataSecPerPoint
    t_between_sweep = raw_data.sweepIntervalSec - raw_data.sweepPointCount*dt
    df_protocol = pd.DataFrame(np.arange(0, t_between_sweep, dt), columns=['t']).set_index('t')
    df_protocol['v'] = waveform.EpochTable(raw_data, 0).holdingLevel

    if save:
        if not new_path:
            new_path = path.split('/')[-1].split('.abf')[0]
            new_path = f'../data/initial_protocols/{new_path}.csv'

        df_protocol.to_csv(new_path)
        print(f'Protocol saved in {new_path} file')      

    return df_protocol

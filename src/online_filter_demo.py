"""
Visualize filtered EEG in real; altered script from original repo
"""

import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop, local_clock
from src.utils.filters import OnlineBandpass
import time
import numpy as np

FS = 250
N_CHANNELS = 8
USE_CHUNK = False #This global variable defines whether we want to use chunking

# Documentation on inlet pylsl: 
# https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/inlet.html#classlsl_1_1stream__inlet_1a558f53812f5dc3c19b2cbe0026a61f6a


def plot(data_zi, data_nozi, latencies, sample_zi):

    plt.ion()
    fig, (ax, ax_nozi) = plt.subplots(2, 1) 

    ax.clear()
    ax.plot(np.array(data_zi)[:, 0])                          # channel 0 only
    if USE_CHUNK:
        ax.set_title("Filtered EEG ch0, zi preserved (chunking)")
    else:
        ax.set_title("Filtered EEG ch0, zi preserved (singular samples)")

    ax_nozi.clear()
    ax_nozi.plot(np.array(data_nozi)[:, 0])                # channel 0 only
    ax_nozi.axvline(max(0, len(data_nozi) - len(sample_zi)), color="r", ls=":")   # where the newest chunk starts

    if USE_CHUNK:
        ax.set_title("Filtered EEG ch0, zi resets at input (chunking)")
    else:
        ax.set_title("Filtered EEG ch0, zi resets at input (singular samples)")

                        
    fig.tight_layout()
    if USE_CHUNK:
        plt.savefig("zi_comparison_chunking.png")
    else:
        plt.savefig("zi_comparison_samples.png")

    plt.close()
    print(f"mean latency: {np.mean(np.array(latencies)) * 1000}")

def experiment():

    print("Staring experiment")
    streams = resolve_byprop("type", "EEG", 1, timeout= 5)
    print(streams[0])
    inlet = StreamInlet(streams[0])
    # time correction
    offset = inlet.time_correction()
    # filter with zi
    filt = OnlineBandpass(FS, 8, 30, N_CHANNELS, use_zi=True)
    # filter without zi
    filt_nozi = OnlineBandpass(FS, 8, 30, N_CHANNELS, use_zi=False)

    data_zi = []
    data_nozi = []
    latencies = []

    if USE_CHUNK:
        pull_func = inlet.pull_chunk
    else:
        pull_func = inlet.pull_sample

    start = time.time()
    end = start + 15

    while time.time() < end:
        sample, timestamp = pull_func()
        if np.size(timestamp) == 0:          
            continue
        
        sample_zi = filt.process(sample)   
        sample_nozi = filt_nozi.process(sample) 

        data_zi.extend(sample_zi) # use extend for chunking
        data_nozi.extend(sample_nozi)

        # in the case of a chunk, there are multiple timestamps
        ts = np.asarray(timestamp)
        ts = ts + offset
        latency = local_clock() - ts
        latencies.extend(np.atleast_1d(latency))

        if len(data_zi) == 250:
            data_zi = data_zi[-1 * FS:]            
            data_nozi = data_nozi[-1 * FS:]
            plot(data_zi, data_nozi, latencies, sample_zi)

            break

experiment()

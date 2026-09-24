"""
Visualize filtered EEG in real 
https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/inlet.html#classlsl_1_1stream__inlet_1a558f53812f5dc3c19b2cbe0026a61f6a
"""

import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop, local_clock
from src.filters import OnlineBandpass
import time
import numpy as np

FS = 250
N_CHANNELS = 8
USE_CHUNKING = True

print("Staring experiment")
streams = resolve_byprop("type", "EEG", 1, timeout= 5)
print(streams[0])
inlet = StreamInlet(streams[0])
# filter with zi
filt = OnlineBandpass(FS, 8, 30, N_CHANNELS, use_zi=True)
# filter without zi
filt_nozi = OnlineBandpass(FS, 8, 30, N_CHANNELS, use_zi=False)

plt.ion()
fig, (ax, ax_nozi) = plt.subplots(2, 1) 

data = []
data_nozi = []
latencies = []

if USE_CHUNKING:
    pull_func = inlet.pull_chunk
else:
    pull_func = inlet.pull_sample

start = time.time()
end = start + 10

while time.time() < end:
    sample, timestamp = pull_func()
    if np.size(timestamp) == 0:          
        continue
    
    sample_nozi = filt_nozi.process(sample) 
    sample = filt.process(sample)

    data.extend(sample) # use extend for chunking
    data_nozi.extend(sample_nozi)

    # in the case of a chunk, there are multiple timestamps
    ts = np.asarray(timestamp)
    latency = local_clock() - ts
    latencies.extend(np.atleast_1d(latency))

    data = data[-2 * FS:]              # show 2 seconds instead of 1
    data_nozi = data_nozi[-2 * FS:]

    ax.clear()
    ax.plot(np.array(data)[:, 0])                          # channel 0 only
    ax.set_title("Filtered EEG ch0 (8–30 Hz), zi preserved")

    ax_nozi.clear()
    ax_nozi.plot(np.array(data_nozi)[:, 0])                # channel 0 only
    ax_nozi.axvline(max(0, len(data_nozi) - len(sample)), color="r", ls=":")   # where the newest chunk starts
    ax_nozi.set_title("Filtered EEG ch0 (8–30 Hz), zi reset every chunk")

    plt.pause(0.5)                     # was 0.01: now each chunk is ~125 samples


    # data = data[-FS:]     
    # data_nozi = data_nozi[-FS:]             

    # ax.clear()
    # ax.plot(data)
    # ax.set_title("Filtered EEG (8–30 Hz), zi preserved")
    # ax_nozi.clear()
    # ax_nozi.plot(data_nozi)
    # ax_nozi.set_title("Filtered EEG (8–30 Hz), zi reset every chunk")
    # plt.pause(0.01)

plt.close()
print(f"mean latency {np.mean(np.array(latencies))}")

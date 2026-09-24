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
USE_ZI = True
USE_CHUNKING = False

print("Staring experiment")
streams = resolve_byprop("type", "EEG", 1, timeout= 5)
print(streams[0])
inlet = StreamInlet(streams[0])
filt = OnlineBandpass(FS, 8, 30, N_CHANNELS, use_zi=USE_ZI)

plt.ion()
fig, ax = plt.subplots()

data = []
latencies = []

if USE_CHUNKING:
    pull_func = inlet.pull_chunk
else:
    pull_func = inlet.pull_sample

start = time.time()
end = start + 60

while time.time() < end:
    sample, timestamp = pull_func()
    if np.size(timestamp) == 0:          
        continue
    sample = filt.process(sample)

    data.extend(sample) # extend for chunking

    # in the case of a chunk, there are multiple timestamps
    ts = np.asarray(timestamp)
    latency = local_clock() - ts
    latencies.extend(np.atleast_1d(latency))

    data = data[-FS:]                  

    ax.clear()
    ax.plot(data)
    ax.set_title("Filtered EEG (8–30 Hz)")
    plt.pause(0.01)

plt.close()
print(f"mean latency {np.mean(np.array(latencies))}")

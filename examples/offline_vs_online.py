"""
Compare offline and online filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from src.filters import OnlineBandpass


FS = 250
N_CHANNELS = 1  

signal = np.random.randn(FS * 2, N_CHANNELS)

b, a = butter(4, [8, 30], fs=FS, btype="band")
offline = filtfilt(b, a, signal[:, 0])

online_filt = OnlineBandpass(FS, 8, 30, N_CHANNELS)
online = [online_filt.process([s])[0] for s in signal[:, 0]]

# https://medium.com/data-science/instantaneous-phase-and-magnitude-with-the-hilbert-transform-40a73985be07
# https://medium.com/@RaghavKrishna25/hilbert-transform-in-action-from-signals-to-insights-206edb017f2f
# https://www.youtube.com/watch?v=7CimsUF8jwI
phase_offline = np.angle(hilbert(offline)) 
phase_online = np.angle(hilbert(online))


phase_difference_rad = np.angle(
    np.mean(
        np.exp(1j * (phase_offline - phase_online))
    )
)

phase_difference_deg = np.degrees(phase_difference_rad)

# phase_difference_deg = np.mean(phase_offline - phase_online)

print(phase_difference_deg)

plt.plot(offline, label="offline (filtfilt)")
plt.plot(online, label="online (causal)")
plt.legend()
plt.title("Offline vs Online Filtering")
plt.show()


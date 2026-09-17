"""
Compare offline and online filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from src.filters import OnlineBandpass
from scipy.signal import correlate

FS = 250
N_CHANNELS = 1  

signal = np.random.randn(FS * 1, N_CHANNELS)

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


# returns an array of the discrete cross-correlation between the two filtered signals
# the length of the cross-correlation is 2*N - 1 because it looks at the shift in both direction
# so it's double the number of samples
xcorr = correlate(online, offline)
print(f"the length of the cross-correlation is {len(xcorr)}")
print(xcorr.argmax())


# n is the number of samples (we could also use FS)
n = len(offline)
# delta time array to match cross-correlation
dt = np.arange(1-n, n)

recovered_time_shift = dt[xcorr.argmax()]



print(f"phase shift leads to a delay at the {recovered_time_shift}th sample")
print(f"the time delay is {recovered_time_shift * 1/250}")

# plt.plot(offline, label="offline (filtfilt)")
# plt.plot(online, label="online (causal)")
# plt.legend()
# plt.title("Offline vs Online Filtering")
# plt.show()


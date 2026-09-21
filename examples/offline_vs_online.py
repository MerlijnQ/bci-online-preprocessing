"""
Compare offline and online filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from src.filters import OnlineBandpass
from scipy.signal import correlate

np.random.seed(42)

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

print(f"the phase shift is {phase_difference_deg} degrees")


# https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves
# https://dsp.stackexchange.com/questions/59036/cross-correlation-dimensions
# https://www.youtube.com/watch?v=0yFdKsv5L-U

# returns an array of the discrete cross-correlation between the two filtered signals
# the length of the cross-correlation is 2*N - 1 because it looks at the shift in both direction
xcorr = correlate(online, offline)


# n is the number of samples (we could also use FS)
n = len(offline)
# delta time array to match cross-correlation
dt = np.arange(1-n, n)


# get the index of the 
recovered_time_shift = dt[xcorr.argmax()]
# time shift in milliseconds
time_shift_seconds = recovered_time_shift * 1/250 * 1000

print(f"the time delay is {time_shift_seconds} milliseconds or {time_shift_seconds / 1000} seconds")

plt.plot(offline, label="offline (filtfilt)")
plt.plot(online, label="online (causal)")
plt.legend()
plt.title("Offline vs Online Filtering")
plt.show()


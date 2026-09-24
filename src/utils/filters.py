"""
Causal EEG filters for real-time use; altered script from original repo
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

class OnlineBandpass:
    """
    Stateful causal band-pass filter
    """
    def __init__(self, fs, low, high, n_channels, order=4, use_zi=True):
        self.b, self.a = butter(
            order, [low, high],
            btype="band", fs=fs
        )
        if use_zi:
            self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, n_channels))
        else:
            self.zi=None

    def process(self, data):
        """
        Process multichannel EEG sample
        """
        # adjusted so it accepts chunks of samples
        x = np.atleast_2d(np.asarray(data, dtype=float))
        if self.zi is None:
            # lfilter returns only y when zi isn't passed
            return lfilter(self.b, self.a, x, axis=0)
        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y


class QuasiCausalFilter:
    """
    Quasi-causal band-pass filter
    """
    def __init__(self, fs, low, high, order=4):
        self.b, self.a = butter(order, [low, high], btype="band", fs=fs)

    def process_window(self, window):
        # use filtfilt (non-causal filter) on the window
        return filtfilt(self.b, self.a, window, axis=0)
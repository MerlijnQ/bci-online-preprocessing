"""
Causal EEG filters for real-time use
"""

import numpy as np
from scipy.signal import butter, lfilter

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

    def process(self, sample):
        """
        Process one multichannel EEG sample
        """
        y, self.zi = lfilter(
            self.b, self.a,
            [sample], axis=0, zi=self.zi
        )
        return y[0]

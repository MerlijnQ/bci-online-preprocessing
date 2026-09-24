"""
Sliding window utilities for streaming data
"""

import numpy as np

class SlidingWindow:
    def __init__(self, size, step, channels):
        self.size = size
        self.step = step
        self.channels = channels
        self.buffer = np.empty((0, channels))
        self.step_counter = 0
        self.is_full = False

    def update(self, data):
        # Accepts a single sample of shape (channels,) 
        # or a chunk of shape (n_samples, channels).
        data = np.asarray(data)

        chunk = np.atleast_2d(data)
        self.buffer = np.vstack((self.buffer, chunk))

        # Check
        if not self.is_full:
            # check it window is full
            if len(self.buffer) >= self.size:
                self.is_full = True
                # when window is full, restart counter
                self.step_counter = 0
                # Return most recent window 
                return self.buffer[-self.size:].T
            return None

        # increase the counter. This also works for chunks of samples
        self.step_counter += len(chunk)

        # if there are too many samples
        if self.step_counter >= self.step:
            self.step_counter = 0
            # Trim excess historical buffer to prevent unbounded growth
            if len(self.buffer) > self.size + self.step:
                self.buffer = self.buffer[-self.size:]
            return self.buffer[-self.size:].T

        return None

# window = SlidingWindow(size = 4, step = 2, channels = 2)

# test = window.update((5,5))
# test = window.update((6,6))
# test = window.update((7,7))
# test = window.update((8,8))
# print(window.step_counter)
# print(test)

"""
Sliding window utilities for streaming data
"""

import numpy as np

class SlidingWindow:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.buffer = []
        self.counter = 0

    def update(self, sample):
        #Fix it to work with multiple channels such that the buffer becomes size (buffer, channels)
        #Fix is to work with a chunk being provided and added
        

        self.buffer.append(sample)

        if len(self.buffer) > self.size:
            self.buffer.pop(0)

        if len(self.buffer) == self.size and self.counter % self.step == 0:
            self.counter += 1
            window = np.array(self.buffer)
            if len(window.shape) > 1:
                return window.T
            else:
                return window

        return None

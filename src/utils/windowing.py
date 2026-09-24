"""
Sliding window utilities for streaming data; altered script from original repo
"""

import numpy as np

class SlidingWindow:
    def __init__(self, size, step, channels):
        self.size = size
        self.step = step
        self.channels = channels
        self.buffer = np.empty((0, channels))
        self.counter = 0
        self.is_full = False

    def update(self, sample):

        sample = np.array(sample)
        self.buffer = np.vstack((self.buffer, sample))

        if self.is_full:
            self.counter += len(sample)

        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size:]      

        if len(self.buffer) == self.size and self.counter % self.step == 0:
       
            self.is_full = True
            
            if len(self.buffer.shape) > 1:
                return self.buffer.T
            else:
                return self.buffer
            

        return None

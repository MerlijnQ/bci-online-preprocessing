"""
Minimal real-time preprocessing pipeline
"""

from pylsl import StreamInlet, resolve_stream
from src.filters import OnlineBandpass
from src.windowing import SlidingWindow
from scipy.signal import butter, filtfilt, hilbert


FS = 250
N_CHANNELS = 8

streams = resolve_stream("type", "EEG")
inlet = StreamInlet(streams[0])

b, a = butter(4, [8, 30], fs=FS, btype="band")

filt = OnlineBandpass(fs=FS, low=8, high=30, n_channels=N_CHANNELS)
window = SlidingWindow(size=FS, step=FS // 4)

while True:
    sample, _ = inlet.pull_sample()
    # sample = filt.process(sample)
    win = window.update(sample) #Needs to return (buffer, channels)
    non_causal_filter = filtfilt(b, a, window)


    if win is not None:
        print("Window ready:", win.shape)

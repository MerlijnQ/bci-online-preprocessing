"""
Minimal real-time preprocessing pipeline; altered script from original repo
"""

from pylsl import StreamInlet, resolve_streams
from src.utils.filters import OnlineBandpass, QuasiCausalFilter
from src.utils.windowing import SlidingWindow
import time

FS = 250
N_CHANNELS = 8
USE_CHUNK = True

streams = resolve_streams(1)
inlet = StreamInlet(streams[0])

# Causal filter
causal_filter = OnlineBandpass(fs=FS, low=8, high=30, n_channels=N_CHANNELS)
causal_window = SlidingWindow(size=FS, step=FS // 4, channels=N_CHANNELS)


# Quasi-causal
quasi_filter = QuasiCausalFilter(fs=FS, low=8, high=30, order=4)
raw_window = SlidingWindow(size=FS, step=FS // 4, channels=N_CHANNELS)


start = time.time()
end = start + 30

while time.time() < end:

    if USE_CHUNK:
        data, _ = inlet.pull_chunk() 
    else:
        data, _ = inlet.pull_sample()  

    raw_win = None
    causal_win = None

    if data is not None and len(data) > 0:
        # Causal filter: first filter, then window
        filtered_data = causal_filter.process(data)
        causal_win = causal_window.update(filtered_data)

        # Quasi-causal filter: first window, then filter
        raw_win = raw_window.update(data)
    
    # Both windows emit at the exact same step interval
    if raw_win is not None and causal_win is not None:
        # Note: SlidingWindow returns (channels, samples) due to .T,
        # transpose back to (samples, channels) for filtfilt along axis=0
        raw_win_time_first = raw_win.T
        quasi_win_time_first = quasi_filter.process_window(raw_win_time_first)

        causal_win_time_first = causal_win.T

        # Inspect the latest sample (rightmost edge: t = now)
        latest_causal = causal_win_time_first[-1, 0]
        latest_quasi = quasi_win_time_first[-1, 0]

        print(
            f"Latest sample ch0 -> Causal: {latest_causal:+.3f} | Quasi: {latest_quasi:+.3f}"
        )

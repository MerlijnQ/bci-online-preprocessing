# bci-online-preprocessing

We altered the scripts that were provided in the original repository. You can find these under src. We moved reusable scripts to a dedicated utils folder. Figures that were created and can be found in the report are also available under the Figures folder. In order to switch between the inlet pulling a chunk or a single sample, a globally defined variable USE_CHUNK needs to be adjusted at the top of the following scripts: pipeline.py and online_filter_demo.


# Original text

This repository focuses on preprocessing EEG signals in real time.
All operations are causal and suitable for streaming applications.

Topics covered in this repository:
- Causal band-pass filtering
- Stateful filter design
- Sliding window segmentation
- Offline vs online comparison

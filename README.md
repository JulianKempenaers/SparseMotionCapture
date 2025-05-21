# SparseMotionCapture

A real-time video capture tool that stores only moving pixels in sparse matrix format to drastically reduce experimental data file sizes.

## Features
- Captures video at 5 frames per second (fps).
- Detects moving pixels by comparing each frame to a key background frame.
- Saves only detected pixels as sparse matrices (BSR format) in compressed .npz files.
- Runs in real-time during video capture.
- Ideal for experimental setups with a static background.


### License
[MIT License](LICENSE)

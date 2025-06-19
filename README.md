# SparseMotionCapture

A real-time video capture tool that stores only moving pixels in sparse matrix format to drastically reduce experimental data file sizes.
- Built in **Python**
- Works with **Picam2 camera** on [**Raspberry Pi**](https://www.raspberrypi.com/) computers.

## Summary
In home surveillance or research setups with a fixed camera, most footage consists of a static background with occasional movement. This tool saves only the moving parts, enabling storage of hours of footage without wasting space on empty frames. This tool runs live in the backgorund while footage is recorded, so there is no need to manually sort through footage afterwards.

## Technical details
### [SparseMotionCapture.py](SparseMotionCapture.py):
- Captures video at max 5 frames per second (fps).
- Periodically saves key frames (Full frames)
- Compares each new frame to the most recent key frame to detect motion.
- Saves only the detected (changing) pixels as sparse matrices (BSR format) in compressed .npz files.
- Runs in real-time during video capture.
- Ideal for experimental setups or surveillance cameras with a static background and fixed camera.

### [NpzToMp4.py](NpzToMp4.py):
- Converts the compressed .npz files back into .mp4 videos for viewing and analysis.
- .mp4 files are larger in size. For long-term storage, keep data in .npz format. Use .mp4 only for temporary playback or review.

### License
[MIT License](LICENSE)

![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Advanced Video Scene Detection and Analysis

**PySceneDetect is a powerful Python tool for automatically detecting scene changes in videos and analyzing video content.** [(See the original repository)](https://github.com/Breakthrough/PySceneDetect)

## Key Features:

*   **Scene Detection:** Accurately identifies scene cuts and transitions in videos using various detection algorithms (ContentDetector, AdaptiveDetector, ThresholdDetector).
*   **Video Splitting:**  Supports splitting videos into individual scenes, leveraging `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Easily saves key frames from each scene for review and analysis.
*   **Python API:** Offers a flexible Python API for seamless integration into video processing pipelines.
*   **Command-Line Interface (CLI):** Provides a user-friendly CLI for quick scene detection, splitting, and image saving.
*   **Configurable:** Allows customization of detection algorithms, thresholds, and output options.
*   **Benchmarking:** Evaluates the performance of different detectors for accuracy and speed.

## Getting Started

### Quick Install

Install PySceneDetect and the necessary dependencies (including OpenCV for certain features) using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

**Note:** Requires `ffmpeg` or `mkvmerge` for video splitting. Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).

### Quick Start (Command Line)

Here are some examples of how to use the CLI:

*   **Split video on cuts:**
    ```bash
    scenedetect -i video.mp4 split-video
    ```
*   **Save images from cuts:**
    ```bash
    scenedetect -i video.mp4 save-images
    ```
*   **Skip the first 10 seconds:**
    ```bash
    scenedetect -i video.mp4 time -s 10s
    ```

For more detailed usage, refer to the [CLI documentation](https://www.scenedetect.com/cli/).

### Quick Start (Python API)

Here's a basic example using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

This will detect scenes using content-aware detection. `scene_list` will contain the start and end times of each scene.

Example to print scene information:
```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

You can also split the video into scenes using `ffmpeg`:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For advanced API usage examples, see [the API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Examples:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)
*   **Benchmark Report:** [benchmark/README.md](benchmark/README.md)

## Contributing & Support

*   **Issue Tracker:**  Report bugs and feature requests on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
*   **Pull Requests:** Contributions are welcome!
*   **Discord:** Get help and discuss PySceneDetect on the [official Discord server](https://discord.gg/H83HbJngk7).
*   **Contact:** Reach out through [my website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
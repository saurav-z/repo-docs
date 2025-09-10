<!-- PySceneDetect Logo -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Powerful Video Cut Detection and Analysis

**PySceneDetect** is a robust and versatile open-source Python tool designed to detect scene changes (cuts, fades, dissolves, etc.) in videos and perform various video analysis tasks. Learn more and contribute at the [official GitHub repository](https://github.com/Breakthrough/PySceneDetect).

<!-- Badges -->
[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

<!-- Release Information -->
**Latest Release: v0.6.7 (August 24, 2023)**

**Key Features:**

*   **Scene Detection:** Automatically identifies scene changes using multiple detection algorithms.
*   **Video Splitting:** Splits videos into individual scenes using ffmpeg or mkvmerge.
*   **Image Extraction:** Saves key frames from each scene for easy preview and analysis.
*   **Python API:** Easily integrate scene detection into your Python workflows.
*   **Command-Line Interface (CLI):** Provides a user-friendly command-line interface for quick and easy video processing.
*   **Adaptive Detection:** Includes adaptive detection for handling fast camera movement.
*   **Fade Detection:** Detects fade in and fade out events.
*   **Highly Configurable:** Offers a wide range of configuration options for precise control.

<!-- Installation -->
## Installation

Install PySceneDetect using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

**Note:** Requires ffmpeg/mkvmerge for video splitting support. Windows builds (MSI installer/portable ZIP) can be found on the [download page](https://scenedetect.com/download/).

<!-- Quick Start - CLI -->
## Quick Start (Command Line)

### Split video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

### Save images from each cut:

```bash
scenedetect -i video.mp4 save-images
```

### Skip the first 10 seconds:

```bash
scenedetect -i video.mp4 time -s 10s
```

Find more examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

<!-- Quick Start - Python API -->
## Quick Start (Python API)

Here's a basic example using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```
You can also split the video into scenes if `ffmpeg` is installed (`mkvmerge` is also supported):

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced usage, explore the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

<!-- Benchmarks -->
## Benchmark

We evaluate the performance of different detectors in terms of accuracy and processing speed. See the [benchmark report](benchmark/README.md) for details.

<!-- Resources -->
## Resources

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

<!-- Contributing -->
## Help & Contributing

Report bugs and suggest features on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues). Before submitting, check existing issues to avoid duplicates.
Pull requests are welcome and encouraged. PySceneDetect is licensed under the BSD 3-Clause license.

For help, join the [Discord Server](https://discord.gg/H83HbJngk7), submit an issue [on Github](https://github.com/Breakthrough/PySceneDetect/issues), or contact me via [my website](http://www.bcastell.com/about/).

<!-- Code Signing -->
## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

<!-- License -->
## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

<!-- Copyright -->
----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
<!-- PySceneDetect Logo -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Powerful Video Cut Detection and Analysis

Detect scene changes and analyze video content effortlessly with PySceneDetect, a versatile Python tool.  ([See the original repo](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release:** v0.6.7 (August 24, 2025)

## Key Features

*   **Scene Detection:** Automatically identify scene changes in videos using various detection algorithms.
*   **Video Splitting:**  Split videos into individual scenes, supporting ffmpeg and mkvmerge.
*   **Frame Extraction:** Save key frames from each detected scene.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows.
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick video analysis and processing.
*   **Adaptive Detection:** Handles fast camera movement effectively.
*   **Fade Detection:**  Detects fade-in and fade-out events.
*   **Highly Configurable:** Customize detection algorithms and output formats.
*   **Benchmark:** Detailed performance evaluation of different detection methods.

## Quick Install

Install PySceneDetect with optional OpenCV support:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting. Windows users can find pre-built installers on the [download page](https://scenedetect.com/download/).

## Quick Start - Command Line

Split a video into scenes using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the video:

```bash
scenedetect -i video.mp4 time -s 10s
```

Explore more CLI examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

## Quick Start - Python API

Detect scenes in a video using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

`scene_list` contains the start and end times of each detected scene.  You can iterate through the scenes:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

Split the video into scenes:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced API usage, see the [documentation](https://www.scenedetect.com/docs/latest/api.html).

## Benchmarks

Evaluate the performance of different detectors in terms of accuracy and speed.  See the [benchmark report](benchmark/README.md) for details.

## Documentation and Resources

*   [Documentation](https://www.scenedetect.com/docs/) (Application and Python API)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Discord:** https://discord.gg/H83HbJngk7

## Contributing & Support

Report bugs, request features, and contribute to the project via the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
Pull requests are welcome!  PySceneDetect is BSD 3-Clause licensed.

For help, join the [Discord Server](https://discord.gg/H83HbJngk7), submit an issue [on GitHub](https://github.com/Breakthrough/PySceneDetect/issues), or contact the author via [the website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
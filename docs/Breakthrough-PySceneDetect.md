<p align="center">
  <img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200"/>
</p>

# PySceneDetect: Intelligent Video Cut Detection and Analysis

**PySceneDetect** is a powerful Python tool for automatically detecting scene changes in videos, providing accurate scene detection and versatile video processing capabilities. ([See the original repo](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release: v0.6.7 (August 24, 2025)**

## Key Features

*   **Accurate Scene Detection:** Employing advanced algorithms for precise identification of scene changes, including fast cuts, fades, and dissolves.
*   **Command-Line Interface (CLI):** Easily process videos from the command line with intuitive and flexible commands.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows with a comprehensive and configurable API.
*   **Video Splitting:** Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from detected scenes for visual analysis.
*   **Content-Aware Detection:** Detect scenes based on visual content, allowing for more intelligent scene recognition.
*   **Multiple Detection Algorithms:** Support for different detection algorithms, including content-based and threshold-based methods.
*   **Benchmarking:** Evaluate the performance of different detectors in terms of accuracy and processing speed.

## Quick Install

Install PySceneDetect with all the necessary dependencies for video splitting (requires `ffmpeg` or `mkvmerge`):

```bash
pip install scenedetect[opencv] --upgrade
```

## Quick Start (Command Line)

Detect scenes and split a video:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each detected scene:

```bash
scenedetect -i video.mp4 save-images
```

## Quick Start (Python API)

Detect scenes and get a list of scene start/end times:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Split the video into scenes using `ffmpeg`:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Examples:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)

## Help & Contributing

Report bugs, request features, and contribute to the project:

*   **Issue Tracker:** [https://github.com/Breakthrough/PySceneDetect/issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Pull Requests:** Welcome!
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
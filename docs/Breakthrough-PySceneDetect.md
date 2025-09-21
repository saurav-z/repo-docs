<!-- PySceneDetect Logo -->
<img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200">

# PySceneDetect: Your Ultimate Video Scene Detection and Analysis Tool

PySceneDetect is a powerful, open-source tool for detecting scene changes in videos, offering both command-line and Python API functionality. **[View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect)**.

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Accurate Scene Detection:** Detects scene changes using various algorithms (Content, Adaptive, and Threshold detectors) to suit different video types.
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick video analysis and splitting.
*   **Python API:** Flexible API for integrating scene detection into your custom workflows and applications.
*   **Video Splitting:** Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from each scene change for visual analysis.
*   **Customizable:** Highly configurable to meet your specific needs, including threshold settings and detection algorithms.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.

## Quick Start

### Installation

Install PySceneDetect with OpenCV support using `pip`:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` and/or `mkvmerge` for video splitting.

### Command-Line Example

Split a video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save a few frames from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds:

```bash
scenedetect -i video.mp4 time -s 10s
```

### Python API Example

Detect scenes and print the scene list:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
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
*   **Benchmark Report:** [benchmark/README.md](benchmark/README.md)

## Contributing & Support

*   **Issue Tracking:**  [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues) - Report bugs, request features, and search existing issues.
*   **Pull Requests:**  Welcome and encouraged!
*   **Contact:** You can join the official PySceneDetect Discord Server, submit an issue/bug report on Github, or contact the maintainer via their website (see README for contact information).

## Code Signing

PySceneDetect utilizes code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

## License

PySceneDetect is licensed under the BSD-3-Clause License. See the `LICENSE` and `THIRD-PARTY.md` files for more details.
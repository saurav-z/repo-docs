[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Advanced Video Scene Detection and Analysis

PySceneDetect is a powerful Python tool for automatically detecting and analyzing scene changes in videos, enabling seamless editing and content management.  Explore the original repository [here](https://github.com/Breakthrough/PySceneDetect)

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Key Features:**

*   **Accurate Scene Detection:**  Uses advanced algorithms, including content and threshold detectors, to identify scene changes effectively.
*   **Command-Line Interface (CLI):**  Offers a user-friendly CLI for quick video analysis and splitting.
*   **Python API:** Provides a flexible Python API for integrating scene detection into custom workflows and applications.
*   **Video Splitting:** Supports splitting videos into individual scenes using `ffmpeg` and `mkvmerge`.
*   **Frame Extraction:**  Saves key frames from detected scenes for visual reference.
*   **Adaptive Detection:** Features a two-pass `AdaptiveDetector` to handle fast camera movements efficiently.
*   **Fade Detection:** Includes `ThresholdDetector` for accurate detection of fade-in and fade-out effects.
*   **Highly Configurable:**  Offers extensive customization options for detection algorithms, video splitting, and more.

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires `ffmpeg` and/or `mkvmerge` for video splitting functionality.*  Find Windows builds (MSI installer/portable ZIP) on [the download page](https://scenedetect.com/download/).

**Quick Start (Command Line):**

Split a video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save scene images:

```bash
scenedetect -i video.mp4 save-images
```

Process a section of the video:

```bash
scenedetect -i video.mp4 time -s 10s
```

More examples are available in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

**Quick Start (Python API):**

Detect scenes using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Iterate through the detected scenes:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

Split video into scenes using ffmpeg:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For detailed API usage, see [the documentation](https://www.scenedetect.com/docs/latest/api.html).

**Benchmark:**

Performance is evaluated in terms of accuracy and speed; see the [benchmark report](benchmark/README.md) for details.

**Reference:**

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

**Help & Contributing:**

Report bugs or request features on [the Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).  Pull requests are welcome.

Join the [Discord Server](https://discord.gg/H83HbJngk7), submit an issue on [Github](https://github.com/Breakthrough/PySceneDetect/issues), or contact the developer via [website](http://www.bcastell.com/about/).

**Code Signing:**

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
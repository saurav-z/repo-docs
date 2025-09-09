# PySceneDetect: Intelligent Video Scene Detection and Analysis

[![PySceneDetect Logo](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

**PySceneDetect is a powerful Python-based tool for detecting scene changes in videos, offering accurate and efficient analysis.**  Visit the [original repository](https://github.com/Breakthrough/PySceneDetect) for more details.

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Key Features:**

*   **Accurate Scene Detection:** Identifies scene changes using advanced algorithms, including content-aware and threshold-based detection.
*   **Command-Line Interface (CLI):** Easily analyze videos from the command line with flexible options for splitting, saving images, and more.
*   **Python API:** Integrate scene detection seamlessly into your Python projects with a comprehensive and configurable API.
*   **Video Splitting Support:** Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Flexible Detection Methods:** Supports ContentDetector, AdaptiveDetector, and ThresholdDetector for various scene change types (cuts, fades, etc.).
*   **Fast Camera Movement Handling:** The AdaptiveDetector handles fast camera movements better.
*   **Frame-Accurate Analysis:** Provides precise start and end times for each scene.
*   **Extensive Documentation:** Comprehensive documentation to help you get started quickly and master advanced features.
*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Benchmarking:** Performance reports available to evaluate accuracy and processing speed

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting. Windows builds (MSI installer/portable ZIP) can be found on the [download page](https://scenedetect.com/download/).

**Quick Start (Command Line):**

Split a video on each fast cut using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save some frames from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the input video:

```bash
scenedetect -i video.mp4 time -s 10s
```

More examples can be found in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

**Quick Start (Python API):**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

The `scene_list` variable will now hold a list containing the start/end times of all scenes found in the video.

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

You can also split the video into scenes if `ffmpeg` or `mkvmerge` are installed:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

See the [documentation](https://www.scenedetect.com/docs/latest/api.html) for detailed API usage.

**Benchmark:**

Evaluate the performance of different detectors in terms of accuracy and processing speed. See the [benchmark report](benchmark/README.md) for details.

**Reference**

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

**Get Help & Contribute:**

Report bugs or suggest features via the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues). Contributions are welcome!  Join the [official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7), or contact me via [my website](http://www.bcastell.com/about/).

**Code Signing:**

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Advanced Scene Detection and Video Analysis

**PySceneDetect is a powerful Python tool designed for accurate scene detection, video splitting, and analysis.**  You can explore the original repository [here](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

---

**Latest Release: v0.6.7 (August 24, 2025)**

**Key Features:**

*   **Accurate Scene Detection:** Identify scene changes using advanced algorithms like ContentDetector, AdaptiveDetector, and ThresholdDetector.
*   **Video Splitting:**  Effortlessly split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Command-Line Interface (CLI):** Simple and intuitive CLI for quick video analysis and manipulation.
*   **Python API:**  Highly configurable Python API for seamless integration into your video processing pipelines.
*   **Frame Extraction:**  Save key frames from detected scenes.
*   **Flexible Configuration:** Customize detection parameters, time ranges, and output formats.
*   **Benchmarking:** Performance evaluation to understand detector accuracy and processing speed.

---

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires ffmpeg/mkvmerge for video splitting support.  Windows builds are available on the [download page](https://scenedetect.com/download/).*

---

**Quick Start (Command Line):**

Detect cuts and split a video:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the input video:

```bash
scenedetect -i video.mp4 time -s 10s
```

Explore more CLI examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

**Quick Start (Python API):**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

`scene_list` will contain the start/end times of the scenes.

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

See the [documentation](https://www.scenedetect.com/docs/latest/api.html) for advanced API usage.

---

**Benchmark:**

For performance details, refer to the [benchmark report](benchmark/README.md).

---

## Resources

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Examples](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues)
*   [Discord](https://discord.gg/H83HbJngk7)

---

## Contribute & Get Help

*   Submit bug reports and feature requests on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
*   Pull requests are welcome.
*   For help, contact the community on [Discord](https://discord.gg/H83HbJngk7).

---

## Code Signing

Uses code signing by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

---

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano. All rights reserved.
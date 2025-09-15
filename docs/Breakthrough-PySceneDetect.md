![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Advanced Video Scene Detection and Analysis

**PySceneDetect** is a powerful, open-source tool for automatically detecting scene changes and analyzing video content.  [Check out the original repo for more details](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release: v0.6.7 (August 24, 2023)**

**Key Features:**

*   **Accurate Scene Detection:** Identify scene changes using various algorithms, including content-based and threshold-based detectors.
*   **Command-Line Interface (CLI):**  Easily detect scenes, split videos, and extract frames from the command line.
*   **Python API:** Integrate scene detection into your Python projects with a flexible and configurable API.
*   **Video Splitting:** Automatically split videos into individual scenes using FFmpeg or mkvmerge.
*   **Customizable Detection:**  Adjust parameters to optimize scene detection for different video types and content.
*   **Fast Camera Movement Detection:** AdaptiveDetector to handle fast camera movement.
*   **Fade In/Out Detection:** ThresholdDetector for handling fade in/out events.
*   **Benchmark Performance:** Evaluate the performance of different detectors in terms of accuracy and processing speed.

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

**Requires:** `ffmpeg` or `mkvmerge` for video splitting.

**Quick Start (Command Line):**

Split a video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save some frames from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the video:

```bash
scenedetect -i video.mp4 time -s 10s
```

For more command-line examples, see the [CLI documentation](https://www.scenedetect.com/cli/).

**Quick Start (Python API):**

Detect scenes and print scene boundaries:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print(f"Scene {i+1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
```

Split a video into scenes using the API:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more API usage, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

**Resources:**

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7

**Help & Contributing:**

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Pull Requests:** Welcome!
*   **Discord:** [Join the Discord Server](https://discord.gg/H83HbJngk7)

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

**Code Signing**

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)
[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

## PySceneDetect: Advanced Video Cut Detection and Analysis

Detect scene changes and analyze video content with PySceneDetect, a powerful and easy-to-use Python tool.  **[Visit the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect) for the latest version and more information.**

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Key Features:**

*   **Accurate Scene Detection:** Identify scene changes using various detection algorithms, including content-aware, adaptive, and threshold-based methods.
*   **Command-Line Interface (CLI):**  Easily analyze and split videos directly from the command line.
*   **Python API:** Integrate PySceneDetect seamlessly into your Python workflows with a flexible and well-documented API.
*   **Video Splitting:**  Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from each scene change for visual analysis.
*   **Highly Configurable:** Customize detection parameters, algorithms, and output formats to fit your needs.
*   **Benchmark Results:** Evaluate the performance of different detectors.

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires `ffmpeg` or `mkvmerge` for video splitting.* Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).

**Quick Start (Command Line Examples):**

*   Split a video based on scene changes:

    ```bash
    scenedetect -i video.mp4 split-video
    ```
*   Save images from each scene cut:

    ```bash
    scenedetect -i video.mp4 save-images
    ```
*   Skip the first 10 seconds of the video during analysis:

    ```bash
    scenedetect -i video.mp4 time -s 10s
    ```

    More examples can be found in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

**Quick Start (Python API Example):**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

**Documentation & Resources:**

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Config File:** [scenedetect.com/docs/0.6.4/cli/config_file.html](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Benchmark:** [benchmark/README.md](benchmark/README.md)

**Help & Contributing:**

Report bugs/issues and request features on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).  Pull requests are welcome.

**Code Signing & License:**

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)
Released under the BSD-3-Clause license; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md).

**Copyright:**

Copyright (C) 2014-2024 Brandon Castellano.  All rights reserved.
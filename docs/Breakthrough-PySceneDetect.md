# PySceneDetect: Powerful Video Scene Detection and Analysis

[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

**PySceneDetect is a versatile Python library designed for accurate and efficient video scene detection, allowing you to analyze, split, and manipulate video content with ease.**

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release: v0.6.7 (August 24, 2025)**

**Key Features:**

*   **Fast and Accurate Scene Detection:** Detect cuts, fades, and other scene changes using various detection algorithms.
*   **Command-Line Interface (CLI):**  Provides a user-friendly CLI for easy video analysis and processing.
*   **Python API:** Integrate PySceneDetect seamlessly into your Python projects for advanced customization and control.
*   **Video Splitting:**  Split videos into individual scenes using FFmpeg or MKVToolNix.
*   **Frame Extraction:** Save representative frames from each scene for preview or analysis.
*   **Flexible Configuration:** Customize detection parameters, including thresholds and detection algorithms.
*   **Content-Aware Detection:**  Utilizes algorithms like the ContentDetector and AdaptiveDetector for optimal results.
*   **Cross-Platform Support:** Works on Windows, macOS, and Linux.

**Getting Started:**

**Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires ffmpeg/mkvmerge for video splitting support.*  Windows builds (MSI installer/portable ZIP) can be found on the [download page](https://scenedetect.com/download/).

**Command-Line Example:**

Split a video into scenes and save frames:

```bash
scenedetect -i video.mp4 split-video save-images
```

More examples are available in the [CLI documentation](https://www.scenedetect.com/cli/).

**Python API Example:**

Detect scenes and print their start/end times:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print(f"Scene {i+1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
```

Find more API examples in the [documentation](https://www.scenedetect.com/docs/latest/api.html).

**Resources:**

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Quickstart:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)
*   **GitHub Repository:** [https://github.com/Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect)

**Help and Contributing:**

*   **Issues:** Submit bug reports and feature requests via the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).  Please search existing issues first.
*   **Pull Requests:**  Contribute to the project with code compliant with the BSD 3-Clause license.
*   **Support:** For help, join the [Discord Server](https://discord.gg/H83HbJngk7), submit an issue on GitHub, or contact the maintainer through [their website](http://www.bcastell.com/about/).

**Benchmark:**

See the [benchmark report](benchmark/README.md) for performance details of different detectors.

**Code Signing:**

This project utilizes code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
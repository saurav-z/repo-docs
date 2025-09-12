# PySceneDetect: Powerful Video Cut Detection and Analysis

[![PySceneDetect Logo](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

**PySceneDetect** is a robust and versatile Python library designed for automatic scene detection in videos, offering a suite of tools for video analysis and processing.

[View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect)

**Key Features:**

*   **Accurate Scene Detection:** Uses advanced algorithms to identify scene changes (cuts, fades, dissolves) in videos.
*   **Command-Line Interface (CLI):** Provides a simple and efficient CLI for easy video processing and analysis.
*   **Python API:** Offers a flexible Python API for seamless integration into custom workflows and applications.
*   **Multiple Detection Algorithms:** Supports various detection methods, including content-aware detection, adaptive detection, and threshold-based detection.
*   **Video Splitting:** Integrates with `ffmpeg` (and `mkvmerge`) to automatically split videos into individual scenes.
*   **Frame Extraction:** Allows saving of key frames from scene changes for visual analysis.
*   **Highly Configurable:**  Offers extensive configuration options to tailor scene detection to specific video content and requirements.
*   **Comprehensive Documentation:** Includes thorough documentation covering the application and Python API.

**Quick Install:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires ffmpeg/mkvmerge for video splitting support. Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).*

**Getting Started:**

**Command Line Example (Split video):**

```bash
scenedetect -i video.mp4 split-video
```

**Python API Example (Detect scenes):**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

**Resources:**

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Quickstart:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Benchmark Report:** [benchmark/README.md](benchmark/README.md)

**Help & Contributing:**

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Pull Requests:** Welcome and encouraged!
*   **Discord Server:**  https://discord.gg/H83HbJngk7
*   **Website:** [http://www.bcastell.com/about/](http://www.bcastell.com/about/)

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

**Code Signing:**

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)
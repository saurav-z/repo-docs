<div align="center">
  <img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200">
</div>

# PySceneDetect: Video Cut Detection and Analysis

**PySceneDetect is a powerful and versatile tool for automatic scene detection and video analysis.** ([See the original repository](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

---

**Latest Release: v0.6.7 (August 24, 2023)**

**Key Features:**

*   **Accurate Scene Detection:** Identify scene changes using content-aware and threshold-based detection algorithms.
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick scene splitting, image saving, and video analysis.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows.
*   **Video Splitting:** Split videos automatically based on detected scenes using `ffmpeg` or `mkvmerge`.
*   **Customizable:** Highly configurable with various detection algorithms and API options.
*   **Frame Extraction:** Extract key frames from scenes for visual analysis.
*   **Performance:** Evaluation of different detectors in terms of accuracy and processing speed.

---

## Quick Start

### Installation

```bash
pip install scenedetect[opencv] --upgrade
```

*Note: Requires ffmpeg/mkvmerge for video splitting support.*  Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).

### Command Line Example

Split input video on each fast cut using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

### Python API Example

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

## Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Quickstart:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Benchmark:** [benchmark/README.md](benchmark/README.md)

## Contributing

*   **Issue Tracker:** [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues)
*   Pull requests are welcome and encouraged.
*   PySceneDetect is released under the BSD 3-Clause license.

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
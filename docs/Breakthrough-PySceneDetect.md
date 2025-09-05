<!--
  SPDX-License-Identifier: BSD-3-Clause
-->
<div align="center">
  <a href="https://github.com/Breakthrough/PySceneDetect">
    <img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="250">
  </a>
</div>

# PySceneDetect: Powerful and Efficient Video Scene Detection

**PySceneDetect is a robust and easy-to-use Python library and command-line tool for automatically detecting scene changes (cuts) in videos.**  This makes it a valuable asset for video editing, analysis, and content management. 

[View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect)

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Accurate Scene Detection:** Employs advanced algorithms to identify cuts, fades, and dissolves in video footage.
*   **Multiple Detection Algorithms:** Offers various scene detection methods (Content, Adaptive, Threshold) to suit different video types and needs.
*   **Command-Line Interface (CLI):** Provides a user-friendly CLI for quick video analysis, scene splitting, and image extraction.
*   **Python API:** Allows seamless integration into your Python workflows for custom video processing and automation.
*   **Video Splitting:** Integrates with `ffmpeg` and `mkvmerge` to automatically split videos into individual scenes.
*   **Frame Extraction:** Enables saving of frames from detected scene changes for visual review or thumbnail creation.
*   **Highly Configurable:** Offers flexibility in parameters such as scene detection thresholds, and timecode settings.
*   **Fast and Efficient:** Optimized for speed and efficiency, making it suitable for large video files and batch processing.
*   **Well-Documented:** Comprehensive documentation ([scenedetect.com/docs/](https://www.scenedetect.com/docs/)) and examples to help you get started quickly.

## Installation

Install PySceneDetect using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

**Important:** Video splitting requires `ffmpeg` or `mkvmerge` to be installed on your system.

## Quick Start Examples

**Command-Line:**

Split a video into scenes using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each scene change:

```bash
scenedetect -i video.mp4 save-images
```

**Python API:**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Config File:** [scenedetect.com/docs/0.6.4/cli/config_file.html](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Benchmark:** [benchmark/README.md](benchmark/README.md)

## Contributing and Support

Contributions are welcome! Please submit bug reports, feature requests, and pull requests via [the Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).

For help or other issues, you can join the official PySceneDetect Discord Server, submit an issue/bug report on Github, or contact me via [my website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

## Copyright

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
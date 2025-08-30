<!-- PySceneDetect Logo (for SEO purposes) -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Scene Detection and Analysis

**PySceneDetect** is a powerful Python library and command-line tool designed for fast and accurate scene detection in video files.  ([View the original repository](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Scene Detection:** Automatically identifies scene changes in videos using various algorithms (Content-aware, Adaptive, Threshold).
*   **Command-Line Interface (CLI):** Simple and intuitive commands for quick scene splitting, image saving, and more.
*   **Python API:**  Integrate scene detection seamlessly into your Python projects.
*   **Video Splitting:**  Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from each scene for easy review or thumbnail generation.
*   **Configurable:** Adjust detection parameters for optimal results with different video content.
*   **Benchmarking:** Performance evaluation to ensure accuracy and speed.

## Quick Installation

Install PySceneDetect using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

**Note:** Requires `ffmpeg` or `mkvmerge` for video splitting support.

## Quick Start (Command Line)

1.  **Split a video on cuts using `ffmpeg`:**

    ```bash
    scenedetect -i video.mp4 split-video
    ```

2.  **Save images from each cut:**

    ```bash
    scenedetect -i video.mp4 save-images
    ```

3.  **Skip the first 10 seconds:**

    ```bash
    scenedetect -i video.mp4 time -s 10s
    ```

    Find more examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

## Quick Start (Python API)

```python
from scenedetect import detect, ContentDetector

scene_list = detect('my_video.mp4', ContentDetector())

for i, scene in enumerate(scene_list):
    print(f'Scene {i+1}: Start {scene[0].get_timecode()} / Frame {scene[0].frame_num}, End {scene[1].get_timecode()} / Frame {scene[1].frame_num}')
```
Further usage in the [documentation](https://www.scenedetect.com/docs/latest/api.html).

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Config File:** [scenedetect.com/docs/0.6.4/cli/config_file.html](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Benchmark Report:** [benchmark/README.md](benchmark/README.md)
*   **Discord:** https://discord.gg/H83HbJngk7

## Contributing and Support

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Pull Requests:** Welcome and encouraged!
*   **Discord Server:** https://discord.gg/H83HbJngk7
*   **Contact:** Via [my website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
<!-- PySceneDetect Logo (Optional) -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Scene Detection and Analysis

**PySceneDetect** is a powerful open-source tool that intelligently detects scene changes in your videos, offering a suite of features for video analysis and editing.  ([View on GitHub](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release:** v0.6.7 (August 24, 2025)

## Key Features

*   **Accurate Scene Detection:**  Identifies scene changes using content-aware and threshold-based detection methods.
*   **Command-Line Interface (CLI):** Easily analyze and process videos with intuitive commands.
*   **Python API:**  Integrate scene detection directly into your Python scripts and workflows.
*   **Video Splitting:**  Automatically split videos into individual scenes using FFmpeg or mkvmerge.
*   **Frame Saving:** Extract key frames from detected scenes for review or analysis.
*   **Customizable:**  Adjust detection parameters and integrate with existing video processing pipelines.
*   **Cross-Platform:** Runs on Windows, macOS, and Linux.
*   **Comprehensive Documentation:**  Detailed documentation and examples for both CLI and API usage.
*   **Scene Analysis:** Offers a tool to calculate shot lengths and other scene statistics

## Quickstart Installation

Install PySceneDetect and the necessary dependencies with the following command:

```bash
pip install scenedetect[opencv] --upgrade
```

*Note:*  Requires FFmpeg/mkvmerge for video splitting support.  See [the download page](https://scenedetect.com/download/) for Windows builds.

## Basic Usage

**Command Line Example:**

Detect scenes and split a video using FFmpeg:

```bash
scenedetect -i video.mp4 split-video
```

**Python API Example:**

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg

scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)

# Iterate through the scene list to access each scene start and end time/frame
for i, scene in enumerate(scene_list):
    print(f"Scene {i+1}: Start {scene[0].get_timecode()} / Frame {scene[0].frame_num}, End {scene[1].get_timecode()} / Frame {scene[1].frame_num}")
```

## Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Quickstart Guide:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **CLI Reference:** [scenedetect.com/docs/latest/cli.html](https://www.scenedetect.com/docs/latest/cli.html)
*   **API Reference:** [scenedetect.com/docs/latest/api.html](https://www.scenedetect.com/docs/latest/api.html)
*   **Benchmark Report:** [benchmark/README.md](benchmark/README.md)

## Contributing and Support

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Website:** http://www.bcastell.com/about/

Contributions are welcome!  Please submit bug reports, feature requests, and pull requests through the issue tracker.

## Code Signing and License

*   **Code Signing:**  Uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).
*   **License:**  BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.
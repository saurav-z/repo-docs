<!-- PySceneDetect Logo -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Scene Detection & Analysis

**PySceneDetect** is a powerful, open-source tool for automatically detecting scene changes and performing video analysis, making video editing and content management a breeze. [View the original repository](https://github.com/Breakthrough/PySceneDetect).

---

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

---

## Key Features

*   **Accurate Scene Detection:** Identify scene changes with precision using advanced algorithms.
*   **Multiple Detection Methods:** Supports Content-aware scene detection, AdaptiveDetector, and ThresholdDetector for various video types.
*   **Command-Line Interface (CLI):** Easily integrate scene detection into your workflows with a simple command-line tool.
*   **Python API:** Powerful Python API for custom integrations and automation.
*   **Video Splitting:** Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Image Saving:** Extract key frames from scenes for easy preview and content selection.
*   **Flexible Timecode Support:** Specify start and end times, and skip sections.
*   **Configurable:** Offers a highly configurable API to suit diverse needs.
*   **Benchmarking:** Evaluate detector performance for optimal results.

---

## Quickstart Installation

Install PySceneDetect with the following command:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` / `mkvmerge` for video splitting support. Windows builds can be found on the [download page](https://scenedetect.com/download/).

---

## Quick Start Examples

### Command Line

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

More examples can be found throughout [the documentation](https://www.scenedetect.com/docs/latest/cli.html).

### Python API

Detect scenes using the ContentDetector:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Iterate and print the detected scenes:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

Split the video using ffmpeg:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more API examples, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

---

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Config File:** [scenedetect.com/docs/0.6.4/cli/config_file.html](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)

---

## Contribute and Get Help

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues) - Submit bug reports, feature requests, or issues.
*   **Pull Requests:** Welcome!  Ensure your contributions adhere to the BSD 3-Clause license.
*   **Help:** Join the [Discord server](https://discord.gg/H83HbJngk7), submit an issue on GitHub, or contact the developer through [my website](http://www.bcastell.com/about/).

---

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

---

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
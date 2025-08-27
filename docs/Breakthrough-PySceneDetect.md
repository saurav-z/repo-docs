[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Powerful Video Scene Detection and Analysis

PySceneDetect is a robust, open-source tool for automatically detecting scene changes and analyzing video content.  

**[View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect)**

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Accurate Scene Detection:** Automatically identify scene changes using various detection algorithms.
*   **Flexible API:** Integrate scene detection into your custom video processing pipelines with a comprehensive Python API.
*   **Command-Line Utility:** Easily process videos from the command line for quick analysis and splitting.
*   **Video Splitting Support:**  Split videos into individual scenes using ffmpeg or mkvmerge.
*   **Content-Aware Detection:**  Detect scene changes based on visual content, not just abrupt cuts.
*   **Frame Extraction:** Save key frames from each detected scene.
*   **Adaptive Scene Detection:** Improved detection with `AdaptiveDetector` to handle fast camera movement.
*   **Fade Detection:** Handles fade in and fade out events.
*   **Benchmarking:** Detailed performance evaluation available [here](benchmark/README.md).

## Quick Installation

Install PySceneDetect with the necessary OpenCV dependencies using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires ffmpeg/mkvmerge for video splitting support. Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).

## Quick Start (Command Line)

Example usage:

*   **Split video:** `scenedetect -i video.mp4 split-video`
*   **Save scene images:** `scenedetect -i video.mp4 save-images`
*   **Process from a specific time:** `scenedetect -i video.mp4 time -s 10s`

More examples available in the [CLI documentation](https://www.scenedetect.com/cli/).

## Quick Start (Python API)

Detect scenes using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Iterate through scenes:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

Split the video:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For advanced usage, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Reference

*   [Documentation](https://www.scenedetect.com/docs/) (application and Python API)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

## Get Help & Contribute

*   **Issue Tracker:** [Report bugs or feature requests](https://github.com/Breakthrough/PySceneDetect/issues).
*   **Pull Requests:**  Contributions are welcome.
*   **Discord:** [Join the official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7).
*   **Contact:**  Reach out via [my website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

---

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
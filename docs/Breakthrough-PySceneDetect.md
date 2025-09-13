![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Cut Detection and Analysis

**PySceneDetect** is a powerful Python tool for automatically detecting scene changes in videos, enabling efficient video analysis and editing. For more information, visit the [original repository](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

**Latest Release:** v0.6.7 (August 24, 2025)

**Key Features:**

*   **Automated Scene Detection:** Accurately identifies scene changes using various detection algorithms.
*   **Command-Line Interface (CLI):** Easily detect and split videos from the command line.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows.
*   **Video Splitting:** Automatically split videos based on detected scene changes using ffmpeg/mkvmerge.
*   **Customizable:** Configure detection parameters and integrate with various video processing pipelines.
*   **Frame Extraction:** Save key frames from detected scenes.
*   **Multiple Detectors:** Includes content-aware (`ContentDetector`), two-pass adaptive (`AdaptiveDetector`), and threshold-based (`ThresholdDetector`).

**Quick Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires ffmpeg/mkvmerge for video splitting support.* Windows builds (MSI installer/portable ZIP) can be found on [the download page](https://scenedetect.com/download/).

**Quick Start (Command Line):**

Split video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the input video:

```bash
scenedetect -i video.mp4 time -s 10s
```

More CLI examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

**Quick Start (Python API):**

Detect scenes and print scene start/end times:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print(f'Scene {i+1}: Start {scene[0].get_timecode()} / Frame {scene[0].frame_num}, End {scene[1].get_timecode()} / Frame {scene[1].frame_num}')
```

Split video using the API:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

Advanced API use in the [documentation](https://www.scenedetect.com/docs/latest/api.html).

**Benchmark:**

Review the performance of detectors in terms of accuracy and processing speed: [benchmark report](benchmark/README.md)

## Resources

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

## Get Help & Contribute

Report bugs or request features via the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).

Contributions are welcome; submit pull requests.  PySceneDetect is released under the BSD 3-Clause license.

For help, or for other questions:
*   Join [the official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7).
*   Submit an issue/bug report [here on Github](https://github.com/Breakthrough/PySceneDetect/issues).
*   Contact via [my website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
<!-- PySceneDetect Logo -->
<img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200">

# PySceneDetect: Powerful Video Scene Detection and Analysis

**PySceneDetect** is a robust and versatile Python tool designed to automatically detect scene changes in videos, providing accurate cut detection and analysis.  Explore the original repository on [GitHub](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Accurate Scene Detection:** Detects scene changes with high precision using content-aware and threshold-based algorithms.
*   **Command-Line Interface (CLI):** Offers a user-friendly CLI for easy video analysis and splitting.
*   **Python API:** Provides a flexible Python API for seamless integration into custom video processing pipelines.
*   **Video Splitting:** Automatically splits videos into individual scenes, with support for `ffmpeg` and `mkvmerge`.
*   **Image Saving:**  Allows saving of key frames from detected scenes.
*   **Fast Camera Movement Handling:** Utilizes an adaptive detector for improved performance with fast camera movements.
*   **Fade Detection:** Includes a `ThresholdDetector` for identifying fade-in and fade-out events.
*   **Configurable:** Offers extensive configuration options for tailoring scene detection to specific video content.

## Installation

Install PySceneDetect with the following command:

```bash
pip install scenedetect[opencv] --upgrade
```

*Note:*  Requires `ffmpeg`/`mkvmerge` for video splitting. Windows builds (MSI installer/portable ZIP) are available on the [download page](https://scenedetect.com/download/).

## Quick Start

### Command Line

Split a video into scenes and save keyframes using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video save-images
```

More CLI examples can be found in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

### Python API

Detect scenes and get a list of scene boundaries:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Example showing scene iteration:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print(f'Scene {i+1}: Start {scene[0].get_timecode()} / Frame {scene[0].frame_num}, End {scene[1].get_timecode()} / Frame {scene[1].frame_num}')
```

Split video into scenes:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced usage, refer to [the API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Benchmark

Evaluate the performance of different detectors in terms of accuracy and processing speed.  See the [benchmark report](benchmark/README.md) for details.

## Documentation and Resources

*   [Documentation](https://www.scenedetect.com/docs/) (Comprehensive Guide)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   [Discord](https://discord.gg/H83HbJngk7)

## Contributing and Support

*   **Issue Tracking:** Report bugs, suggest features, or ask questions on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
*   **Pull Requests:**  Contributions are welcome! Ensure your code adheres to the BSD 3-Clause license.
*   **Discord:**  Get help and chat with the community on the [official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
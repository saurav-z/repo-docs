# PySceneDetect: Powerful Video Scene Detection and Analysis

**Detect cuts, analyze scenes, and split videos effortlessly with PySceneDetect, a versatile Python tool.** ([Original Repository](https://github.com/Breakthrough/PySceneDetect))

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

*   **Latest Release:** v0.6.7 (August 24, 2025)
*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **Discord:** https://discord.gg/H83HbJngk7

## Key Features:

*   **Accurate Scene Detection:** Identify scene changes using various detection algorithms.
*   **Video Splitting:** Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows.
*   **Command-Line Interface (CLI):** Easily process videos from the command line for quick analysis and splitting.
*   **Content-Aware Detection:** Includes `ContentDetector`, `AdaptiveDetector`, and `ThresholdDetector` for handling different video types.
*   **Frame Saving:** Save representative frames from each scene.
*   **Highly Configurable:** Customize detection parameters, output formats, and more.
*   **Benchmarked Performance:** See the [benchmark report](benchmark/README.md) for details on accuracy and speed.

## Quick Installation:

Install PySceneDetect with the `opencv` extra for video processing features:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting support. Windows users can find installers on [the download page](https://scenedetect.com/download/).

## Quick Start (Command Line):

Split a video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save frames from each scene:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds:

```bash
scenedetect -i video.mp4 time -s 10s
```

More examples can be found in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

## Quick Start (Python API):

Detect scenes and get a list of start/end times:

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

Split the video into scenes:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For advanced usage, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Reference:

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

## Help & Contributing:

*   **Issue Tracker:** [https://github.com/Breakthrough/PySceneDetect/issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Discord:** [https://discord.gg/H83HbJngk7](https://discord.gg/H83HbJngk7)

Contributions are welcome. Please submit pull requests.  PySceneDetect is released under the BSD 3-Clause license.

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
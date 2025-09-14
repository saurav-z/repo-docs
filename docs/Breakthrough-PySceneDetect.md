# PySceneDetect: Intelligent Video Cut Detection and Scene Analysis

Easily detect scene changes and analyze video content with PySceneDetect, a powerful and versatile Python tool.  [Check out the original repo](https://github.com/Breakthrough/PySceneDetect)

[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)]()

Key Features:

*   **Accurate Scene Detection:** Identifies scene changes in videos using advanced algorithms.
*   **Multiple Detection Methods:** Supports content-aware (fast cut), fade-in/fade-out, and adaptive scene detection.
*   **Command-Line Interface (CLI):** Provides an intuitive command-line interface for easy video processing.
*   **Python API:** Enables seamless integration into custom Python workflows and applications.
*   **Video Splitting:** Splits videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Saves key frames from detected scenes for visual analysis.
*   **Flexible Configuration:** Highly configurable, allowing for customization of detection parameters.
*   **Benchmarking:** Performance is evaluated and documented in terms of accuracy and processing speed.

## Installation

Install PySceneDetect with `pip`:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg`/`mkvmerge` for video splitting support. Windows builds (MSI installer/portable ZIP) are available on [the download page](https://scenedetect.com/download/).

## Quick Start

### Command Line

Split a video on each fast cut using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save frames from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds of the input video:

```bash
scenedetect -i video.mp4 time -s 10s
```

More examples are available in [the documentation](https://www.scenedetect.com/docs/latest/cli.html).

### Python API

Detect scenes in a video using the Python API:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

`scene_list` will contain the start/end times of all detected scenes.  You can split videos as follows:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced usage, refer to [the API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Reference

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

## Help & Contributing

Report bugs and request features through [the Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
Pull requests are welcome.  PySceneDetect is released under the BSD 3-Clause license.

For help or other issues:
*   Join [the official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7)
*   Submit an issue/bug report [here on Github](https://github.com/Breakthrough/PySceneDetect/issues)

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
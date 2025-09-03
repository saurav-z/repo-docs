<!-- PySceneDetect Logo -->
![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Cut Detection and Analysis

**PySceneDetect** is a powerful, open-source Python tool designed to automatically detect scene changes in your videos, enabling seamless video editing and analysis.  For the latest updates, visit the [original PySceneDetect repository](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Accurate Scene Detection:** Identifies scene changes using advanced algorithms.
*   **Content-Aware Detection:** Detects scene changes based on visual content, including fast cuts, fades, and dissolves.
*   **Command-Line Interface (CLI):** Provides an easy-to-use CLI for quick video analysis and processing.
*   **Python API:** Offers a flexible API for custom integration into your video processing workflows.
*   **Video Splitting:** Supports splitting videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:**  Allows saving representative frames from detected scenes.
*   **Adaptive Detection:** Includes adaptive scene detection to better handle camera movement.
*   **Fade Detection:** Detects fade in/out events.
*   **Configurable:** Highly customizable through a Python API.
*   **Benchmarking:** Performance evaluated for accuracy and speed.

## Installation

```bash
pip install scenedetect[opencv] --upgrade
```

*Note: Requires ffmpeg/mkvmerge for video splitting support.* Windows builds (MSI installer/portable ZIP) are available on the [download page](https://scenedetect.com/download/).

## Quickstart

### Command Line

Split an input video on each fast cut using `ffmpeg`:

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

More examples can be found in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

### Python API

Detect scenes with a high-level function:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Iterate through detected scenes:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

Split the video into scenes using `ffmpeg`:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For advanced usage, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Documentation and Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Quickstart Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **CLI Example:** [CLI Example](https://www.scenedetect.com/cli/)
*   **Config File:** [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   **Benchmark:** [Benchmark Report](benchmark/README.md)

## Contributing and Support

*   **Issue Tracking:** Report bugs and request features on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).  Search existing issues first to avoid duplicates.
*   **Pull Requests:**  Contributions are welcome!  Please adhere to the BSD 3-Clause license.
*   **Help:** Join [the official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7), or contact the author via [the website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
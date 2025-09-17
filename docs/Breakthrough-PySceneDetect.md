![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Video Cut Detection and Analysis Tool

**PySceneDetect** is a powerful open-source tool for automatically detecting scene changes and performing video analysis, allowing for efficient video editing, content analysis, and more. Learn more on the [official GitHub repo](https://github.com/Breakthrough/PySceneDetect).

## Key Features

*   **Scene Detection:** Accurately identifies cuts and scene changes in videos.
*   **Multiple Detection Algorithms:** Includes `ContentDetector`, `AdaptiveDetector`, and `ThresholdDetector` to handle various video types and editing styles.
*   **Command-Line Interface (CLI):**  Easy-to-use CLI for quick scene detection, image extraction, and video splitting.
*   **Python API:**  Highly configurable API for seamless integration into your Python workflows and pipelines.
*   **Video Splitting Support:** Integrated support using `ffmpeg` and `mkvmerge` to split videos into individual scenes.
*   **Frame Extraction:** Saves key frames from detected scenes for easy review and editing.
*   **Flexible Input:** Supports various video formats and allows for time-based processing (e.g., skipping initial seconds).
*   **Benchmarking:** Performance evaluated in terms of accuracy and processing speed.
*   **Cross-Platform:** Available for Windows, macOS, and Linux.

## Installation

Install PySceneDetect using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

**Note:** Requires `ffmpeg`/`mkvmerge` for video splitting support. Windows builds (MSI installer/portable ZIP) are available on the [download page](https://scenedetect.com/download/).

## Quick Start

### Command Line

Split a video into scenes using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save frames from each cut:

```bash
scenedetect -i video.mp4 save-images
```

Skip the first 10 seconds:

```bash
scenedetect -i video.mp4 time -s 10s
```

Find more CLI examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

### Python API

Detect scenes in a video:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Iterate through the scenes:

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

More API examples available in the [documentation](https://www.scenedetect.com/docs/latest/api.html).

## Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Benchmark:** [benchmark/README.md](benchmark/README.md)

## Contributing and Support

*   **Issue Tracker:**  Report bugs and request features on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
*   **Pull Requests:** Welcome and encouraged!
*   **Discord:** Get help and discuss PySceneDetect on the [official Discord Server](https://discord.gg/H83HbJngk7).
*   **Website:** Contact the maintainer via [website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
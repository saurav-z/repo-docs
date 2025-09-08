<p align="center">
  <img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200"/>
</p>

# PySceneDetect: Powerful Video Scene Detection and Analysis

**PySceneDetect** is a robust tool for automatically detecting scene changes in videos, offering a complete suite of analysis and editing capabilities.  Find out more on the [PySceneDetect GitHub Repository](https://github.com/Breakthrough/PySceneDetect).

## Key Features

*   **Accurate Scene Detection:** Utilizes advanced algorithms (ContentDetector, AdaptiveDetector, ThresholdDetector) to identify scene changes with high precision.
*   **Command-Line Interface (CLI):** Easily process videos with straightforward commands for splitting, saving images, and more.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows with a flexible and configurable API.
*   **Video Splitting:**  Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from detected scene changes for preview and analysis.
*   **Customizable Detection:** Configure detection parameters, such as the threshold for scene changes, to tailor the tool to your needs.
*   **Performance Benchmarking:** Evaluate the performance of different detectors to optimize your scene detection process ([Benchmark Report](benchmark/README.md)).

## Installation

Install PySceneDetect with the following command, including the optional `opencv` dependency:

```bash
pip install scenedetect[opencv] --upgrade
```

Ensure you have `ffmpeg` or `mkvmerge` installed for video splitting support. Windows users can find pre-built installers on the [download page](https://scenedetect.com/download/).

## Quick Start Examples

### Command Line

Split a video into scenes:

```bash
scenedetect -i video.mp4 split-video
```

Save frames from each scene:

```bash
scenedetect -i video.mp4 save-images
```

### Python API

Detect scenes in a video:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Split a video into scenes (requires ffmpeg):

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced usage and API examples, refer to [the documentation](https://www.scenedetect.com/docs/latest/api.html).

## Resources

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Examples](https://www.scenedetect.com/cli/)
*   [Configuration File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   [Discord Server](https://discord.gg/H83HbJngk7)

## Contributing and Support

Report bugs, request features, and contribute to the project via [the Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues). Pull requests are welcome.

For help, join the [official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7), submit an issue on GitHub, or contact the maintainer via [the website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

PySceneDetect is licensed under the BSD-3-Clause license. See [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
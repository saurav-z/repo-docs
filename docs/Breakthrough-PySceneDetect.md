<!-- PySceneDetect Logo -->
[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Intelligent Video Scene Detection and Analysis

**PySceneDetect** is a powerful Python library and command-line tool for automatically detecting scene changes and analyzing video content.

**[View the PySceneDetect Repository on GitHub](https://github.com/Breakthrough/PySceneDetect)**

## Key Features

*   **Accurate Scene Detection:** Utilizes advanced algorithms to identify scene changes, including fast cuts, fades, and dissolves.
*   **Command-Line Interface (CLI):** Easily detect scenes, save images, and split videos with simple commands.
*   **Python API:**  Integrate scene detection directly into your Python workflows with a flexible and configurable API.
*   **Multiple Detectors:** Choose from various detection algorithms, including content-aware, adaptive, and threshold-based methods.
*   **Video Splitting Support:**  Split videos automatically using `ffmpeg` or `mkvmerge` based on detected scene changes.
*   **Frame Extraction:** Save key frames from each scene for preview or further analysis.
*   **Detailed Documentation:** Comprehensive documentation covering CLI usage, API examples, and advanced configuration.

## Installation

Install PySceneDetect with `pip`:

```bash
pip install scenedetect[opencv] --upgrade
```

**Note:** Requires `ffmpeg` or `mkvmerge` for video splitting.

## Quick Start

**Command-Line Usage:**

Split a video into scenes using `ffmpeg`:

```bash
scenedetect -i video.mp4 split-video
```

Save images from each scene:

```bash
scenedetect -i video.mp4 save-images
```

**Python API Usage:**

Detect scenes using the content-aware detector:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Split the video using the detected scenes (requires `ffmpeg`):

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more examples, consult the [PySceneDetect Documentation](https://www.scenedetect.com/docs/).

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Examples:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Discord:** [Join our Discord Server](https://discord.gg/H83HbJngk7)

## Contributing

Contributions are welcome!  Please submit bug reports, feature requests, and pull requests via [the GitHub issue tracker](https://github.com/Breakthrough/PySceneDetect/issues).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

PySceneDetect is released under the BSD-3-Clause License.  See the [`LICENSE`](LICENSE) file for details.
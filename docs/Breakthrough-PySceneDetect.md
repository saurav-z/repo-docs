[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Advanced Video Scene Detection & Analysis

**PySceneDetect is a powerful, open-source Python tool that analyzes videos to automatically detect and split scenes, offering robust content-aware scene detection and video editing capabilities.**  [View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect).

[![Build Status](https://img.shields.io/github/actions/workflow/status/Breakthrough/PySceneDetect/build.yml)](https://github.com/Breakthrough/PySceneDetect/actions)
[![PyPI Status](https://img.shields.io/pypi/status/scenedetect.svg)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI Version](https://img.shields.io/pypi/v/scenedetect?color=blue)](https://pypi.python.org/pypi/scenedetect/)
[![PyPI License](https://img.shields.io/pypi/l/scenedetect.svg)](https://scenedetect.com/copyright/)

## Key Features

*   **Automatic Scene Detection:** Identify scene changes based on various algorithms, including content-aware analysis.
*   **Flexible Detection Algorithms:** Choose from multiple detection methods (e.g., ContentDetector, AdaptiveDetector, ThresholdDetector) to suit different video types.
*   **Video Splitting:** Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from each detected scene.
*   **Command-Line Interface (CLI):** Easily use PySceneDetect from the command line for quick video processing.
*   **Python API:** Integrate PySceneDetect into your Python projects for more advanced customization and automation.
*   **Cross-Platform Support:** Works on Windows, macOS, and Linux.
*   **Comprehensive Documentation:** Detailed documentation and examples available at [scenedetect.com/docs/](https://www.scenedetect.com/docs/).

## Quick Installation

Install PySceneDetect using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

*Requires `ffmpeg` or `mkvmerge` for video splitting support.*

## Quick Start Examples

### Command Line

Split a video on fast cuts:

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

### Python API

Detect scenes and print the scene list:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
print(scene_list)
```

Split the video into scenes using `ffmpeg`:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

## Usage & Examples

For detailed usage instructions, API documentation, and more examples, see [scenedetect.com/docs/](https://www.scenedetect.com/docs/).

## Benchmark

Evaluate the performance of different detectors: [benchmark report](benchmark/README.md)

## Reference

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

## Support & Contributing

*   **Issue Tracker:** Report bugs or request features on the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
*   **Discord:** Get help and discuss PySceneDetect on the [Discord Server](https://discord.gg/H83HbJngk7).
*   **Contributions:**  Pull requests are welcome and encouraged.

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.
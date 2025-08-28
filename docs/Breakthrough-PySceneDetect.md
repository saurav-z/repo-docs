# PySceneDetect: Powerful Video Scene Detection and Analysis

[![PySceneDetect Logo](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

**PySceneDetect** is a versatile Python library and command-line tool designed for accurate and efficient video scene detection, analysis, and splitting.

*   [GitHub Repository](https://github.com/Breakthrough/PySceneDetect)
*   [Website](https://scenedetect.com)
*   [Documentation](https://scenedetect.com/docs/)
*   [Discord](https://discord.gg/H83HbJngk7)

**Key Features:**

*   **Scene Detection:** Automatically identifies scene changes (cuts) in videos using various detection algorithms (content-aware, threshold, adaptive).
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick video analysis and splitting.
*   **Python API:** Powerful and flexible Python API for seamless integration into your workflows.
*   **Video Splitting:**  Splits videos into individual scenes using FFmpeg or mkvmerge.
*   **Frame Extraction:** Saves key frames from detected scenes.
*   **Configurable:** Offers extensive customization options for detection parameters.
*   **Benchmarking:** Performance evaluation reports available for different detectors.

**Installation:**

```bash
pip install scenedetect[opencv] --upgrade
```

Requires FFmpeg/mkvmerge for video splitting. Windows builds are available on the [download page](https://scenedetect.com/download/).

**Quick Start Examples:**

**CLI:**

Split video on cuts and save images:

```bash
scenedetect -i video.mp4 split-video save-images
```

**Python API:**

Detect scenes in a video:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

Split a video into scenes:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced usage, explore the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

**Reference:**

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)

**Help & Contributing:**

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Discord:** [Official PySceneDetect Discord Server](https://discord.gg/H83HbJngk7)
*   Pull requests are welcome!

**Code Signing:**

This project uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect) and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect).

**License:**

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.
[![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)](https://github.com/Breakthrough/PySceneDetect)

# PySceneDetect: Advanced Video Scene Detection and Analysis

**PySceneDetect is a powerful Python-based tool for automated video scene detection, analysis, and manipulation.** ([View the original repository](https://github.com/Breakthrough/PySceneDetect))

## Key Features

*   **Accurate Scene Detection:** Identify scene changes with various detection algorithms including content-aware (fast cuts), adaptive, and threshold-based methods.
*   **Command-Line Interface (CLI):** Easily analyze and process videos directly from your terminal.
*   **Python API:** Integrate scene detection seamlessly into your Python workflows and custom applications.
*   **Video Splitting:** Automatically split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Image Saving:** Extract and save key frames from scene changes for easy preview and review.
*   **Flexible Configuration:** Customize detection parameters, output formats, and processing options to fit your needs.
*   **Cross-Platform Support:** Compatible with Windows, macOS, and Linux.
*   **Detailed Documentation:** Comprehensive documentation and examples to get you started quickly.

## Getting Started

**Quick Install:**

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting.

**Example Usage (Command Line):**

```bash
scenedetect -i video.mp4 split-video
scenedetect -i video.mp4 save-images
scenedetect -i video.mp4 time -s 10s
```

**Example Usage (Python API):**

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())

for i, scene in enumerate(scene_list):
    print(f"Scene {i+1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
```

For more detailed examples and API usage, see the [documentation](https://www.scenedetect.com/docs/).

## Resources

*   **Website:** [scenedetect.com](https://www.scenedetect.com)
*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Examples:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7

## Contributing and Support

*   **Issue Tracker:** [GitHub Issues](https://github.com/Breakthrough/PySceneDetect/issues)
*   **Pull Requests:** Welcome!
*   **License:** BSD-3-Clause (see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md))

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.

----------------------------------------------------------

Copyright (C) 2014-2024 Brandon Castellano.
All rights reserved.
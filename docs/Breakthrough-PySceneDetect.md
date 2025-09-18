![PySceneDetect](https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png)

# PySceneDetect: Intelligent Video Cut Detection and Analysis

**PySceneDetect** is a powerful, open-source tool for automatically detecting and analyzing scene changes in your videos, making video editing and processing easier.

[View the original repository on GitHub](https://github.com/Breakthrough/PySceneDetect)

## Key Features

*   **Scene Detection:** Accurately identifies cuts, fades, and dissolves using advanced algorithms.
*   **Command-Line Interface (CLI):** Easily process videos from the command line with intuitive commands.
*   **Python API:** Integrate scene detection directly into your Python scripts and workflows.
*   **Multiple Detectors:** Choose from various detection methods like content-aware, adaptive, and threshold-based.
*   **Video Splitting:** Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   **Frame Extraction:** Save key frames from each scene for easy review and editing.
*   **Highly Configurable:** Customize detection parameters, processing options, and output formats to fit your needs.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Benchmark Report:** Performance evaluated for accuracy and speed.

## Quick Install

Install PySceneDetect with `pip`:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting support. Windows builds (MSI installer/portable ZIP) are available on the [download page](https://scenedetect.com/download/).

## Quick Start (Command Line)

1.  **Split video on cuts using `ffmpeg`:**

    ```bash
    scenedetect -i video.mp4 split-video
    ```

2.  **Save frames from cuts:**

    ```bash
    scenedetect -i video.mp4 save-images
    ```

3.  **Skip the first 10 seconds:**

    ```bash
    scenedetect -i video.mp4 time -s 10s
    ```

    Explore more examples in the [documentation](https://www.scenedetect.com/docs/latest/cli.html).

## Quick Start (Python API)

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
```

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].frame_num,
        scene[1].get_timecode(), scene[1].frame_num,))
```

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

For more advanced use, see the [API documentation](https://www.scenedetect.com/docs/latest/api.html).

## Reference

*   [Documentation](https://www.scenedetect.com/docs/)
*   [CLI Example](https://www.scenedetect.com/cli/)
*   [Config File](https://www.scenedetect.com/docs/0.6.4/cli/config_file.html)
*   [Benchmark Report](benchmark/README.md)

## Help & Contributing

Report bugs and suggest features via the [Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).
Pull requests are welcome.

For support, join the [Discord Server](https://discord.gg/H83HbJngk7), submit an issue/bug report [here on Github](https://github.com/Breakthrough/PySceneDetect/issues), or contact the author via [website](http://www.bcastell.com/about/).

## Code Signing

This program uses free code signing provided by [SignPath.io](https://signpath.io?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect), and a free code signing certificate by the [SignPath Foundation](https://signpath.org?utm_source=foundation&utm_medium=github&utm_campaign=PySceneDetect)

## License

BSD-3-Clause; see [`LICENSE`](LICENSE) and [`THIRD-PARTY.md`](THIRD-PARTY.md) for details.
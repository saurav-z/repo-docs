<div align="center">
  <a href="https://github.com/Breakthrough/PySceneDetect">
    <img src="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png" alt="PySceneDetect Logo" width="200">
  </a>
  <br>
  <h1>PySceneDetect: Intelligent Video Scene Detection and Analysis</h1>
</div>

PySceneDetect is a powerful and versatile tool for automatically detecting scene changes in videos, enabling you to split, analyze, and process your video content with ease.

**[Explore the PySceneDetect Repository](https://github.com/Breakthrough/PySceneDetect)**

---

## Key Features

*   🎬 **Scene Detection:** Accurately identify scene changes using various detection algorithms (Content, Adaptive, Threshold).
*   ✂️ **Video Splitting:**  Split videos into individual scenes using `ffmpeg` or `mkvmerge`.
*   🖼️ **Image Extraction:** Save key frames from each scene for easy review and thumbnail generation.
*   ⚙️ **Highly Configurable:** Customize detection parameters, algorithms, and output formats to suit your specific needs.
*   💻 **Command-Line Interface (CLI):**  Easy-to-use CLI for quick video processing tasks.
*   🐍 **Python API:** Integrate scene detection seamlessly into your Python workflows.
*   ⏱️ **Fast and Efficient:** Optimized for speed and accuracy, with benchmark reports available.

---

## Installation

Install PySceneDetect with all optional dependencies (including OpenCV) using pip:

```bash
pip install scenedetect[opencv] --upgrade
```

Requires `ffmpeg` or `mkvmerge` for video splitting support. Windows builds (MSI installer/portable ZIP) are available on [the download page](https://scenedetect.com/download/).

---

## Quickstart Examples

### Command Line

Split a video into scenes and save images:

```bash
scenedetect -i video.mp4 split-video save-images
```

Skip the first 10 seconds:

```bash
scenedetect -i video.mp4 time -s 10s
```

### Python API

Detect scenes and print scene start/end times:

```python
from scenedetect import detect, ContentDetector
scene_list = detect('my_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print(f"Scene {i+1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
```

Split a video using detected scenes:

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
scene_list = detect('my_video.mp4', ContentDetector())
split_video_ffmpeg('my_video.mp4', scene_list)
```

Refer to the [documentation](https://www.scenedetect.com/docs/) for more detailed examples and advanced usage.

---

## Resources

*   **Documentation:** [scenedetect.com/docs/](https://www.scenedetect.com/docs/)
*   **CLI Example:** [scenedetect.com/cli/](https://www.scenedetect.com/cli/)
*   **Discord:** https://discord.gg/H83HbJngk7
*   **Issue Tracker:** [github.com/Breakthrough/PySceneDetect/issues](https://github.com/Breakthrough/PySceneDetect/issues)

---

## Contributing

Contributions are welcome! Please submit any bugs/issues or feature requests to [the Issue Tracker](https://github.com/Breakthrough/PySceneDetect/issues).  Pull requests are encouraged and should comply with the BSD 3-Clause license.

---

## License

PySceneDetect is released under the BSD 3-Clause License. See the [`LICENSE`](LICENSE) file for details.

---
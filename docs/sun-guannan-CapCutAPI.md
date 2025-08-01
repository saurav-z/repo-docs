# CapCutAPI: Unleash the Power of CapCut with Python

**Unlock advanced video editing capabilities with CapCutAPI, a powerful open-source Python tool for programmatic control and automation. [View the original repository](https://github.com/sun-guannan/CapCutAPI).**

## Key Features

*   **Draft Management:** Create, read, modify, and save CapCut draft files.
*   **Comprehensive Material Support:** Add and edit videos, audios, images, text, and stickers.
*   **Extensive Effects Library:** Implement transitions, filters, masks, and animations.
*   **API-Driven Control:** Access and automate video editing functions via HTTP APIs.
*   **AI Integration:** Leverage AI services for intelligent subtitle generation, text creation, and image enhancements.
*   **Cross-Platform Compatibility:** Works with both CapCut China and International versions.
*   **Automated Workflows:** Enable batch processing and automated video editing.
*   **Flexible Configuration:** Customize functionality with easy-to-use configuration files.

## Quick Start

### Installation

1.  **Configure the project:**
    ```bash
    cp config.json.example config.json
    ```
    Modify `config.json` as needed.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the environment:**
    *   Install `ffmpeg` and add it to system's environment variables.
    *   Ensure Python version 3.8.20 is installed.

4.  **Run the Server:**
    ```bash
    python capcut_server.py
    ```

    Access the API interfaces through the specified ports.

## Usage Examples

### Adding a Video

```python
import requests

response = requests.post("http://localhost:9001/add_video", json={
    "video_url": "http://example.com/video.mp4",
    "start": 0,
    "end": 10,
    "width": 1080,
    "height": 1920
})

print(response.json())
```

### Adding Text

```python
import requests

response = requests.post("http://localhost:9001/add_text", json={
    "text": "Hello, World!",
    "start": 0,
    "end": 3,
    "font": "ZY_Courage",
    "font_color": "#FF0000",
    "font_size": 30.0
})

print(response.json())
```

### Saving a Draft

```python
import requests

response = requests.post("http://localhost:9001/save_draft", json={
    "draft_id": "123456",
    "draft_folder": "your capcut draft folder"
})

print(response.json())
```
### Copying the Draft to CapCut Draft Path

Calling `save_draft` will generate a folder starting with `dfd_` in the current directory of the server. Copy this folder to the CapCut draft directory, and you will be able to see the generated draft.

### More Examples
For more comprehensive usage examples, please refer to the `example.py` file.

## Gallery

### AI-Generated Content Integration

**Explore how CapCutAPI seamlessly integrates AI-generated content.**

[![Horse](https://img.youtube.com/vi/IF1RDFGOtEU/hqdefault.jpg)](https://www.youtube.com/watch?v=IF1RDFGOtEU)

[![Song](https://img.youtube.com/vi/rGNLE_slAJ8/hqdefault.jpg)](https://www.youtube.com/watch?v=rGNLE_slAJ8)

## API Endpoints

*   `/create_draft`: Create a new draft.
*   `/add_video`: Add video material.
*   `/add_audio`: Add audio material.
*   `/add_image`: Add image material.
*   `/add_text`: Add text to the draft.
*   `/add_subtitle`: Add subtitles.
*   `/add_effect`: Add effects to materials.
*   `/add_sticker`: Add stickers.
*   `/save_draft`: Save the current draft.
# CapCutAPI: Open-Source Python API for CapCut Video Editing

Unlock the power of CapCut video editing programmatically with CapCutAPI, an open-source Python tool that allows you to automate and customize your video creation workflow.  [View the original repository on GitHub](https://github.com/sun-guannan/CapCutAPI).

## Key Features

*   **Draft File Management**: Create, read, modify, and save CapCut draft files.
*   **Comprehensive Material Support**: Add and edit videos, audios, images, text, stickers, and more.
*   **Extensive Effect Application**: Apply transitions, filters, masks, and animations to your videos.
*   **Robust API Service**: Utilize HTTP API interfaces for remote calls and automated processing.
*   **AI Integration**: Enhance your videos with AI-powered features like subtitle generation and text/image creation.
*   **Cross-Platform Compatibility:** Supports both CapCut China and International versions.
*   **Automated Workflows:** Enables batch processing and automated video creation pipelines.
*   **Flexible Configuration:** Customize the tool using configuration files.

## Core API Endpoints

*   `/create_draft`: Create a new CapCut draft.
*   `/add_video`: Add video material to a draft.
*   `/add_audio`: Add audio material to a draft.
*   `/add_image`: Add image material to a draft.
*   `/add_text`: Add text to your video drafts.
*   `/add_subtitle`: Add subtitles to your video.
*   `/add_effect`: Apply effects to your video elements.
*   `/add_sticker`: Add stickers to your video drafts.
*   `/save_draft`: Save your finalized draft files.

## Getting Started

### Configuration

1.  **Configuration File:** Customize settings by copying `config.json.example` to `config.json` and modifying the values.
    ```bash
    cp config.json.example config.json
    ```

### Environment Setup

1.  **FFmpeg:** Ensure FFmpeg is installed and accessible in your system's environment variables.
2.  **Python:** Requires Python version 3.8.20.
3.  **Dependencies:** Install project dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

Execute the following command to launch the CapCutAPI server:

```bash
python capcut_server.py
```

You can then access the API endpoints to begin your automated video editing processes.

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

### Integrating with CapCut

Saving a draft generates a folder starting with `dfd_` in the server's directory. Copy this folder into your CapCut draft directory to view and edit the generated draft within the CapCut application.

### Advanced Examples

Refer to the `example.py` file in the project for more advanced API usage, including adding audio and applying effects.

## Demo Videos

*   **AI Cut**:  [![AI Cut](https://img.youtube.com/vi/fBqy6WFC78E/hqdefault.jpg)](https://www.youtube.com/watch?v=fBqy6WFC78E)
*   **Connect AI generated**:  [![Airbnb](https://img.youtube.com/vi/1zmQWt13Dx0/hqdefault.jpg)](https://www.youtube.com/watch?v=1zmQWt13Dx0)
    [![Horse](https://img.youtube.com/vi/IF1RDFGOtEU/hqdefault.jpg)](https://www.youtube.com/watch?v=IF1RDFGOtEU)
    [![Song](https://img.youtube.com/vi/rGNLE_slAJ8/hqdefault.jpg)](https://www.youtube.com/watch?v=rGNLE_slAJ8)
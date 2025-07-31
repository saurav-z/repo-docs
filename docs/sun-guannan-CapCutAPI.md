# CapCutAPI: The Open-Source Python Tool for Editing and Automating CapCut Video Creation

<p>Effortlessly automate your video editing workflow and unlock creative potential with the CapCutAPI, a powerful open-source Python-based tool.  Explore the capabilities and see it in action with video demonstrations.</p>

[View the original repository on GitHub](https://github.com/sun-guannan/CapCutAPI)

## Key Features

CapCutAPI empowers you to manipulate and automate CapCut video projects with a range of powerful features:

*   **Draft File Management:** Create, read, modify, and save CapCut draft files.
*   **Rich Media Support:** Add and edit videos, audio, images, text, and stickers.
*   **Effect Application:** Incorporate transitions, filters, masks, and animations.
*   **API Service:** Utilize HTTP APIs for remote calls and automated processing.
*   **AI Integration:** Integrate AI services for generating subtitles, text, and images.
*   **Cross-Platform Compatibility:** Works with both CapCut China and International versions.
*   **Automated Workflows:** Support batch processing for efficient video creation.
*   **Flexible Configuration:** Customize functionality using configuration files.

## Video Examples

*   **Connecting AI Generated Content:** [View Video](https://www.youtube.com/watch?v=IF1RDFGOtEU)
*   **Adding Audio and Sound Effects:** [View Video](https://www.youtube.com/watch?v=rGNLE_slAJ8)

## Getting Started

### Configuration

1.  **Copy Configuration File:**
    ```bash
    cp config.json.example config.json
    ```
    Modify `config.json` to tailor settings to your needs.

### Environment Setup

1.  **FFmpeg:** Ensure FFmpeg is installed and accessible via your system's environment variables.
2.  **Python:** Install Python 3.8.20.
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Run the Server

1.  Execute the following command to start the server:
    ```bash
    python capcut_server.py
    ```
    Access API functions through the provided interfaces once the server is running.

## Main API Interfaces

*   `/create_draft`: Create a new CapCut draft.
*   `/add_video`: Add video to the draft.
*   `/add_audio`: Add audio to the draft.
*   `/add_image`: Add an image to the draft.
*   `/add_text`: Add text to the draft.
*   `/add_subtitle`: Add subtitles to the draft.
*   `/add_effect`: Add effects to the draft.
*   `/add_sticker`: Add stickers to the draft.
*   `/save_draft`: Save the current draft.

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

For more detailed examples, please refer to the `example.py` file.

### Copying Drafts to CapCut

Saving a draft creates a folder starting with `dfd_`. Copy this folder to your CapCut draft directory to view and use the generated content.

### Testing with REST Client

You can use the `rest_client_test.http` file with a REST Client IDE plugin for easy HTTP testing.

## Project Features at a Glance

*   **Cross-platform support**:  Compatible with both CapCut China and International versions.
*   **Automated Processing**: Supports batch processing and automated workflows.
*   **Rich APIs**: Provides comprehensive API interfaces.
*   **Flexible Configuration**: Easily customizable through configuration files.
*   **AI Enhancement**: Integrates AI services for improved video production efficiency.
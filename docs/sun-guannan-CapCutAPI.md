# CapCutAPI: Open Source Python Tool for Automated Video Editing

**CapCutAPI** is a powerful, open-source Python tool that allows you to programmatically create, edit, and manipulate CapCut video projects, offering a seamless way to automate your video creation workflow.  You can find the original repository here: [https://github.com/sun-guannan/CapCutAPI](https://github.com/sun-guannan/CapCutAPI).

[Try It: https://www.capcutapi.top](https://www.capcutapi.top)

[中文说明](https://github.com/sun-guannan/CapCutAPI/blob/main/README-zh.md)

## Key Features of CapCutAPI

*   **Draft Management:** Create, read, modify, and save CapCut draft files programmatically.
*   **Material Handling:** Easily add and edit video, audio, images, text, and sticker elements.
*   **Extensive Effects:** Implement a wide array of effects, including transitions, filters, masks, and animations.
*   **HTTP API for Automation:** Expose HTTP API interfaces to integrate with remote systems and automate video processing workflows.
*   **AI Integration:** Enhance your videos with built-in AI services for intelligent subtitle generation, text creation, and image processing.
*   **Cross-Platform Compatibility:** Supports both CapCut China and International versions.
*   **Batch Processing and Automation:** Streamline your workflow with automated processing capabilities.
*   **Flexible Customization:** Configure and tailor the tool to your specific needs through configuration files.

## Project Gallery: Showcase of Capabilities

**AI-powered Video Creation:**

*   **AI Cut:** [![AI Cut](https://img.youtube.com/vi/fBqy6WFC78E/hqdefault.jpg)](https://www.youtube.com/watch?v=fBqy6WFC78E)

**Examples of AI-Generated Content Integration:**

*   **Airbnb:** [![Airbnb](https://img.youtube.com/vi/1zmQWt13Dx0/hqdefault.jpg)](https://www.youtube.com/watch?v=1zmQWt13Dx0)
*   **Horse:** [![Horse](https://img.youtube.com/vi/IF1RDFGOtEU/hqdefault.jpg)](https://www.youtube.com/watch?v=IF1RDFGOtEU)
*   **Song:** [![Song](https://img.youtube.com/vi/rGNLE_slAJ8/hqdefault.jpg)](https://www.youtube.com/watch?v=rGNLE_slAJ8)

## Core Functionality & API Endpoints

### Core Features

*   **Draft File Management:** Create, read, modify, and save CapCut draft files.
*   **Material Processing:** Add and edit videos, audios, images, texts, stickers, etc.
*   **Effect Application:** Add transitions, filters, masks, animations, and more.
*   **API Service:** Provide HTTP API interfaces for remote calls.
*   **AI Integration:** Integrate AI for subtitles, texts, and images.

### Main API Interfaces

*   `/create_draft`: Create a new draft.
*   `/add_video`: Add video material to the draft.
*   `/add_audio`: Add audio material to the draft.
*   `/add_image`: Add image material to the draft.
*   `/add_text`: Add text material to the draft.
*   `/add_subtitle`: Add subtitles to the draft.
*   `/add_effect`: Apply effects to materials.
*   `/add_sticker`: Add stickers to the draft.
*   `/save_draft`: Save the draft file.

## Getting Started: Installation and Configuration

### Configuration

Customize the tool's behavior using a configuration file:

1.  **Copy Configuration:**  Duplicate `config.json.example` to `config.json`.
2.  **Modify Settings:**  Adjust the values in `config.json` to suit your requirements.

```bash
cp config.json.example config.json
```

### Prerequisites

*   **ffmpeg:** Ensure ffmpeg is installed on your system and accessible via your system's environment variables.
*   **Python:**  Python 3.8.20 or higher is required.
*   **Dependencies:** Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Running the Server

Launch the CapCutAPI server with the following command:

```bash
python capcut_server.py
```

After the server starts, you can access API functionalities via the defined HTTP endpoints.

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

You can also use the `rest_client_test.http` file with a REST Client IDE plugin for testing.

### Locating the Draft in CapCut

Saving a draft generates a folder named `dfd_*` in the server's directory.  Copy this folder to your CapCut draft directory to view the generated project within CapCut.

### More Examples

Refer to the `example.py` file in the project for additional usage scenarios, including adding audio and effects.
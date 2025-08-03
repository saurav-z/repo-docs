# CapCut API: Automate Video Editing with Python

**Effortlessly create and manipulate CapCut video projects programmatically using this powerful Python-based API.**  [Explore the original repository](https://github.com/sun-guannan/CapCutAPI).

## Key Features

*   **Comprehensive Draft Management:**
    *   Create, read, modify, and save CapCut draft files.
*   **Rich Material Support:**
    *   Add and edit videos, audios, images, text, and stickers.
*   **Extensive Effects Library:**
    *   Apply transitions, filters, masks, and animations.
*   **Robust API Service:**
    *   Utilize HTTP APIs for remote calls and automated processing.
*   **AI-Powered Enhancements:**
    *   Integrate AI for subtitle generation, text creation, and image manipulation.
*   **Cross-Platform Compatibility:**
    *   Supports both CapCut China and International versions.
*   **Automated Workflows:**
    *   Supports batch processing for efficient video editing.
*   **Flexible Configuration:**
    *   Customize functionality using configuration files.

## Showcase

*   **AI-Powered Integration**
    *   [AI Cut](https://www.youtube.com/watch?v=fBqy6WFC78E)
    *   [Airbnb](https://www.youtube.com/watch?v=1zmQWt13Dx0)
    *   [Horse](https://www.youtube.com/watch?v=IF1RDFGOtEU)
    *   [Song](https://www.youtube.com/watch?v=rGNLE_slAJ8)

## Getting Started

### Configuration

1.  **Create Configuration File:**
    *   Copy `config.json.example` to `config.json`.
    *   Modify the configuration settings in `config.json` as needed.

    ```bash
    cp config.json.example config.json
    ```

### Prerequisites

*   **ffmpeg:**  Ensure ffmpeg is installed and accessible in your system's environment variables.
*   **Python:** Python 3.8.20 or higher is required.

### Installation

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

1.  **Start the Server:**

    ```bash
    python capcut_server.py
    ```

    Access the API interfaces after the server starts.

## API Endpoints

*   `/create_draft`: Create a draft
*   `/add_video`: Add video material to the draft
*   `/add_audio`: Add audio material to the draft
*   `/add_image`: Add image material to the draft
*   `/add_text`: Add text material to the draft
*   `/add_subtitle`: Add subtitles to the draft
*   `/add_effect`: Add effects to materials
*   `/add_sticker`: Add stickers to the draft
*   `/save_draft`: Save the draft file

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

### Testing with REST Client

You can also use the `rest_client_test.http` file with a REST Client plugin in your IDE for easy testing.

## Copying Drafts to CapCut

Saving a draft generates a folder starting with `dfd_` in the server's directory. Copy this folder into your CapCut draft directory to view and edit the generated project.
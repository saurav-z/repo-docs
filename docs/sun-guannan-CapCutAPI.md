# CapCutAPI: Automate Video Editing with Python and AI

Enhance your video editing workflow with CapCutAPI, a powerful open-source Python tool that lets you programmatically create, modify, and automate CapCut projects. ([Original Repository](https://github.com/sun-guannan/CapCutAPI))

## Key Features

*   **Draft Management:** Create, read, modify, and save CapCut draft files.
*   **Rich Media Support:** Add and edit videos, audio, images, text, and stickers.
*   **Effect Application:** Integrate transitions, filters, masks, and animations.
*   **HTTP API:** Access features remotely for automated processing.
*   **AI Integration:** Utilize AI for generating subtitles, text, and images, streamlining video creation.
*   **Cross-Platform Compatibility:** Works with both CapCut China and International versions.
*   **Automated Workflows:** Supports batch processing and automated video editing processes.
*   **Flexible Configuration:** Customizable through configuration files.

## Demo & Examples

### Project Showcase

**MCP Agent**
[![AI Cut](https://img.youtube.com/vi/fBqy6WFC78E/hqdefault.jpg)](https://www.youtube.com/watch?v=fBqy6WFC78E)

**Connect AI generated via CapCutAPI**
[![Airbnb](https://img.youtube.com/vi/1zmQWt13Dx0/hqdefault.jpg)](https://www.youtube.com/watch?v=1zmQWt13Dx0)

[![Horse](https://img.youtube.com/vi/IF1RDFGOtEU/hqdefault.jpg)](https://www.youtube.com/watch?v=IF1RDFGOtEU)

[![Song](https://img.youtube.com/vi/rGNLE_slAJ8/hqdefault.jpg)](https://www.youtube.com/watch?v=rGNLE_slAJ8)

### Usage Examples

*   **Adding a Video:**

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

*   **Adding Text:**

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

*   **Saving a Draft:**

    ```python
    import requests

    response = requests.post("http://localhost:9001/save_draft", json={
        "draft_id": "123456",
        "draft_folder": "your capcut draft folder"
    })

    print(response.json())
    ```

## API Endpoints

*   `/create_draft`: Create a new draft.
*   `/add_video`: Add video material.
*   `/add_audio`: Add audio material.
*   `/add_image`: Add image material.
*   `/add_text`: Add text to the draft.
*   `/add_subtitle`: Add subtitles.
*   `/add_effect`: Apply visual effects.
*   `/add_sticker`: Add stickers.
*   `/save_draft`: Save the completed draft.

## Getting Started

### Configuration

1.  **Copy Configuration:** Duplicate `config.json.example` to `config.json` and adjust settings as needed.

    ```bash
    cp config.json.example config.json
    ```

### Prerequisites

*   **ffmpeg:** Ensure ffmpeg is installed and in your system's PATH.
*   **Python:** Requires Python 3.8.20 or later.
*   **Dependencies:** Install project dependencies using pip.

    ```bash
    pip install -r requirements.txt
    ```

### Run the Server

*   Execute the following command to launch the CapCutAPI server:

    ```bash
    python capcut_server.py
    ```

*   Access the API via the defined endpoints.
*   Use the `rest_client_test.http` file with a REST Client IDE plugin for testing.

### Draft Location

*   Saving a draft generates a folder starting with `dfd_` in the current directory.
*   Move this folder into your CapCut draft directory to view the edited draft within CapCut.

### Additional Examples

*   Explore `example.py` for detailed usage examples, including adding audio and effects.

## Contributing

Contributions are welcome!  Please feel free to submit pull requests or open issues.

## Links

*   **Project Source:** [https://github.com/sun-guannan/CapCutAPI](https://github.com/sun-guannan/CapCutAPI)
*   **Try It:** [https://www.capcutapi.top](https://www.capcutapi.top)
*   **Chinese Documentation:** [https://github.com/sun-guannan/CapCutAPI/blob/main/README-zh.md](https://github.com/sun-guannan/CapCutAPI/blob/main/README-zh.md)
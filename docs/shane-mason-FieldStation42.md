# FieldStation42: Relive the Golden Age of Television 📺

**FieldStation42 is a Python-based cable and broadcast TV simulator that brings the nostalgic experience of classic over-the-air television to life.**  [View on GitHub](https://github.com/shane-mason/FieldStation42)

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

## Key Features

*   **Multiple Channels:** Simulate a complete TV lineup with simultaneous channel support.
*   **Realistic Schedules:** Automatically generates weekly schedules based on customizable station configurations.
*   **Commercial & Bump Integration:** Seamlessly integrates commercial breaks and station IDs into content.
*   **Flexible Scheduling:** Supports various channel types, including traditional networks, movie channels, and looping community channels.
*   **Content Management:** Manages video catalogs, ensuring shows play in sequence as if broadcast live.
*   **UI & Web Control:**  Includes a user interface for catalog and schedule management, plus a web-based remote control.
*   **On-Screen Display:** Features a customizable on-screen display showing channel info and the current time.
*   **Hardware Integration:** (Optional) Supports hardware connections for channel changing.
*   **Preview/Guide Channel:**  Provides a channel for previews and information.
*   **Date Range Functionality:** Set date ranges for specific shows like sports or holiday specials.
*   **Looping channels** useful for community bulletin style channels or information loops.

## 🚀 Getting Started

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place video files in the `catalog/` directory.
4.  **Configure Stations:**  Copy an example config from `confs/examples/` to `confs/` and customize it.
5.  **Build Catalogs and Schedules:**
    ```bash
    python3 station_42.py --rebuild_catalog --schedule
    ```
6.  **Start the Player:**
    ```bash
    python3 field_player.py
    ```
7.  **(Optional) Start the Web Server:**
    ```bash
    python3 station_42.py --server
    ```

For a comprehensive guide, refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## 📁 Project Structure

*   `station_42.py` — Main CLI and UI for building catalogs and schedules
*   `field_player.py` — Main TV interface/player
*   `fs42/` — Core Python modules (catalog, schedule, API, etc.)
*   `confs/` — Station and system configuration files
*   `catalog/` — Your video content, organized by channel (created by installer)
*   `runtime/` — Runtime files, sockets, and status (created by installer)
*   `fs42/fs42_server/static/` — Web UI static files (HTML, JS, CSS)
*   `docs/` — Images and documentation

## 🛠️ Configuration & Administration

### Quickstart Setup

*   Ensure Python 3 and MPV are installed.
*   Clone the repository.
*   Run the install script.
*   Add your video content.
*   Configure your stations by copying an example JSON file from `confs/examples` into `confs/`.
*   Generate a weekly schedule using `python3 station_42.py`. Use `--rebuild_catalog` if the video content has changed.
*   Run `field_player.py` to start watching.
*   (Optional) Configure start-on-boot using `fs42/hot_start.sh`.

For detailed steps, consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## How It Works

FieldStation42 recreates the TV experience with multiple components:

*   **`station_42.py`:** Manages content catalogs and generates schedules.
*   **`field_player.py`:** The main TV interface, playing videos based on the schedule and current time.  It uses the runtime/play_status.socket to provide information.

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request.
4.  Ask questions or discuss on the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## 🐞 Troubleshooting

*   **Player won't start:**  Check video paths and config files.
*   **No video/audio:** Verify MPV installation and functionality.
*   **Web UI not loading:** Ensure the server is running with `--server` and check for browser errors.
*   **Database errors:** Check file permissions and Python version.
*   See the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## 📚 Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)
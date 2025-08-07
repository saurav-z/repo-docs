# FieldStation42: Relive the Golden Age of Television 📺

FieldStation42 is a Python-based cable and broadcast TV simulator, offering an authentic experience of watching over-the-air television, complete with channel surfing and continuous programming, allowing you to curate your own retro TV experience. Check out the original repo: [https://github.com/shane-mason/FieldStation42](https://github.com/shane-mason/FieldStation42).

## ✨ Key Features

*   **Multiple Simultaneous Channels:** Simulate a full cable lineup.
*   **Seamless Programming:** Shows continue playing when you switch channels, as if they've been on the air all along.
*   **Automated Scheduling:** Generates weekly schedules based on configurable station settings, including commercials and program bumps.
*   **Flexible Content:** Supports movie-length content.
*   **Dynamic Playlists:** Randomly selects shows to keep the lineup fresh.
*   **Content Management:** Integrates date ranges for seasonal content.
*   **Customization:** Per-station configuration, including sign-off videos and off-air loops.
*   **User-Friendly Interface:** Manage catalogs and schedules through a built-in UI or command-line arguments.
*   **Optional Hardware Integration:** Connect to real-world hardware for channel control.
*   **Web-Based Remote Control:** Control your TV experience remotely.
*   **On-Screen Display:** Customizable channel name, time, and date displays.
*   **Looping Channels:** Ideal for community bulletin boards or information channels.
*   **Preview Channel:** A guide channel with embedded video and configurable messages.

## 🚀 Quick Start Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place your video files in the appropriate folders (see `catalog/` and `confs/examples/`).
4.  **Configure Stations:** Copy an example configuration file from `confs/examples` to `confs/` and edit it to your liking.
5.  **Build Catalogs and Schedules:**
    ```bash
    python3 station_42.py --rebuild_catalog --schedule
    ```
6.  **Launch the Player:**
    ```bash
    python3 field_player.py
    ```
7.  **(Optional) Start the Web Server:**
    ```bash
    python3 station_42.py --server
    ```

For detailed instructions and advanced configuration, consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## 📁 Project Structure

*   `station_42.py`: Main CLI and UI for managing catalogs and schedules.
*   `field_player.py`: The core TV interface/player.
*   `fs42/`: Core Python modules (catalog, schedule, API, etc.).
*   `confs/`: Station and system configuration files.
*   `catalog/`: Your video content, organized by channel.
*   `runtime/`: Runtime files, sockets, and status.
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS).
*   `docs/`: Images and documentation.

## 🛠️ Installation & Setup

### Quickstart Steps

1.  **Prerequisites:** Ensure Python 3 and MPV are installed.
2.  **Clone the Repository:** This will be your main working directory.
3.  **Run the Installer:** Executes the install script.
4.  **Add Content:** Populate with video files.
5.  **Configure Stations:** Copy and modify configuration files.
6.  **Generate Schedule:** Use `station_42.py` to generate a weekly schedule. Use the `--rebuild_catalog` option if your content changes.
7.  **Start Watching:** Run `field_player.py`.
8.  **(Optional) Enable Autostart:** (Not recommended unless a dedicated device is used.) Use `fs42/hot_start.sh` to start on boot.

## 💻 How It Works

FieldStation42 uses several interacting components to emulate a TV broadcast environment.

### `station_42.py`
This script is responsible for building catalogs and generating schedules.  It inspects files on disk to create catalogs;  this can take time depending on the number of videos.  Use this to manage catalogs and schedules, or you can perform all operations using command line arguments with no UI. Use `station_42.py --help` for a full list of options.

### `field_player.py`
This is the main TV interface. It reads the schedule on startup, and opens the video and jumps to the correct time based on the current time. The player writes status and current channel to `runtime/play_status.socket`. See [this page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for more information on integrating with `channel.socket` and `play_status.socket`.

## 🔌 Connecting to a TV & Remote Control

Connect the player to a TV using HDMI (or an HDMI adapter for older TVs). Integrate a remote control using `channel.socket` and `play_status.socket` or, for details on setting up a Bluetooth remote control, see [this page in the discussions](https://github.com/shane-mason/FieldStation42/discussions/47).

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests if possible.
3.  Open a pull request describing your changes.
4.  For questions, use the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## 🐞 Troubleshooting

*   **Player Won't Start:** Verify video file paths and config files.
*   **No Video/Audio:** Confirm MPV is installed and functional.
*   **Web UI Issues:** Ensure the server is running (`--server`) and check for browser errors.
*   **Database Errors:** Check file permissions and Python version.
*   For further assistance, refer to the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## 📚 Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## ⚠️ Alpha Software

This is an early-stage project and requires some technical familiarity:

*   Basic Linux command-line skills
*   JSON configuration file editing
*   Video file management
# FieldStation42: Relive the Golden Age of Television

Bring the nostalgia of classic cable and broadcast TV to life with FieldStation42, a Python-based simulator that recreates the authentic experience of watching over-the-air (OTA) television. [See the original project on GitHub](https://github.com/shane-mason/FieldStation42).

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

---

## 📺 Key Features

*   **Multiple Channels:** Simulate multiple channels playing simultaneously.
*   **Realistic Scheduling:** Generates weekly schedules with commercials and program bumpers.
*   **Content Variety:** Supports feature-length movies and customizable content.
*   **Dynamic Content Selection:** Randomly selects shows to maintain a fresh viewing experience.
*   **Date-Based Scheduling:** Set date ranges for shows (e.g., holiday specials).
*   **Customizable Station Branding:** Configure station sign-offs and off-air loops.
*   **User-Friendly Interface:** Manage catalogs and schedules via the built-in UI.
*   **Optional Hardware Integration:** Connect hardware to control channels.
*   **Web-Based Remote Control:** Use a web interface for remote control functionality.
*   **On-Screen Display:** Channel name, time, and date overlays with customizable icons.
*   **Looping Channels:** Create channels for community bulletins or information loops.
*   **Preview/Guide Channel:** A new feature with embedded video and configurable messages.
*   **Flexible Scheduling:** Designed for traditional and commercial-free channels.

---

## 🚀 Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place your video files in the `catalog/` folders as described in the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).
4.  **Configure Stations:** Copy an example configuration from `confs/examples/` to `confs/` and edit.
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

For detailed setup and configuration, consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## 📁 Project Structure

*   `station_42.py` - Main CLI and UI for catalog/schedule management.
*   `field_player.py` - The main TV interface/player.
*   `fs42/` - Core Python modules (catalog, schedule, API, etc.).
*   `confs/` - Station and system configuration files.
*   `catalog/` - Your video content, organized by channel.
*   `runtime/` - Runtime files, sockets, and status.
*   `fs42/fs42_server/static/` - Web UI static files (HTML, JS, CSS).
*   `docs/` - Images and documentation.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3
*   MPV media player

For a full, step-by-step guide, see the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

### Quickstart Overview

*   Ensure Python 3 and MPV are installed.
*   Clone the repository.
*   Run the install script.
*   Add your content.
*   Configure your stations using the example config files.
*   Generate a weekly schedule using `station_42.py`.
*   Start the TV interface using `field_player.py`.

---

## ⚙️ How It Works

FieldStation42 uses multiple components to recreate a nostalgic TV experience:

### `station_42.py`
This script creates catalogs with metadata about the video content and then generates schedules based on station configuration. You'll typically run this command with `--rebuild_catalog` if you change your media library.  Run this script with no arguments to open a UI for schedule and catalog management, or use command line arguments.

### `field_player.py`
The main TV interface. This script starts by reading the generated schedule and then plays the appropriate video, at the correct position, for the current time. If you switch back to a previous channel, the player picks up where it should be in the program.

The player writes its status to `runtime/play_status.socket` and `runtime/channel.socket`.  See the [wiki](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for details on controlling the player from external applications.

---

## 🔌 Connecting Devices

The `field_player.py` can receive external commands, and the `play_status.socket` and `channel.socket` can be read by other applications. For information on using this system, see the [wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script). You can connect a remote control or other device.

---

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests if possible.
3.  Open a pull request.
4.  For questions, use the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

---

## 🐞 Troubleshooting

*   **Player won't start:** Check video paths and config files.
*   **No video/audio:** Ensure MPV is installed and working.
*   **Web UI not loading:** Verify the server is running with `--server`. Check the browser console for errors.
*   **Database errors:** Check file permissions and Python version.
*   For more help, consult the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or create an issue.

---

## 📚 Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

---

## Disclaimer: Alpha Software

FieldStation42 is in active development, and setup requires some technical knowledge.  You'll need experience with the Linux command line, editing JSON configuration files, and managing video files.
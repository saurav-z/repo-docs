# FieldStation42: Relive the Golden Age of Television

**FieldStation42 is a Python-based software that simulates the experience of watching over-the-air (OTA) television, complete with channel surfing, scheduled programming, and nostalgic retro vibes.**  [Explore the original repo](https://github.com/shane-mason/FieldStation42).

[![Retro TV with Antenna](docs/retro-tv.png?raw=true)](https://github.com/shane-mason/FieldStation42)

## Key Features

*   **Multiple Channel Support:** Simulate a complete cable lineup with various channels running simultaneously.
*   **Realistic Programming:** Enjoy automatically scheduled programming with commercial breaks, bumpers, and custom content.
*   **Dynamic Scheduling:** Generate weekly schedules based on station configurations, including date ranges for seasonal shows.
*   **Content Variety:**  Support for feature-length content like movies and TV shows.  Randomly selects shows to keep the lineup fresh.
*   **Customization:** Configure station sign-off videos, off-air loops, and looping channels for community bulletins.
*   **User-Friendly Interface:** Manage catalogs and schedules through a built-in web-based remote control.
*   **On-Screen Display (OSD):** Displays channel name, time, and date with customizable icons.
*   **Preview/Guide Channel:** Includes embedded video and configurable messages (documentation in progress).
*   **Flexible Channel Types:** Supports traditional network channels with commercials, commercial-free channels, and looping channels for community information.
*   **Hardware Integration:**  Supports optional hardware connections for channel changing.

## Quick Start

Get up and running with your own retro TV simulator in a few steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Content:** Place your video files in the appropriate folders within `catalog/` (refer to `confs/examples/` for structure).
4.  **Configure Stations:**  Copy an example config file from `confs/examples` to `confs/` and modify it to match your desired channels and programming.
5.  **Build Catalogs & Schedules:**
    ```bash
    python3 station_42.py --rebuild_catalog --schedule
    ```
6.  **Start the Player:**
    ```bash
    python3 field_player.py
    ```
7.  **(Optional) Run Web Server:**
    ```bash
    python3 station_42.py --server
    ```

For more detailed instructions and advanced configuration options, please consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## Project Structure

*   `station_42.py`: Main CLI and UI for building catalogs and schedules
*   `field_player.py`:  The main TV interface/player application.
*   `fs42/`:  Core Python modules (catalog, schedule, API, etc.)
*   `confs/`:  Station and system configuration files.
*   `catalog/`:  Your video content, organized by channel (created by installer).
*   `runtime/`:  Runtime files, sockets, and status (created by installer).
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS)
*   `docs/`:  Images and documentation.

## Installation & Setup

For a comprehensive setup guide, refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

### Quickstart Setup Summary

1.  **Prerequisites:** Ensure Python 3 and MPV are installed.
2.  **Clone:** Clone the repository (your working directory).
3.  **Install:** Run the provided installation script.
4.  **Content:** Add your video content to the appropriate directories.
5.  **Configure:**  Edit station configuration files (JSON format).
6.  **Schedule:** Generate a weekly schedule using the command-line tool.
7.  **Watch TV:** Run the `field_player.py` script.
8.  **(Optional)** Configure to start automatically (not recommended unless dedicated device.)

## How It Works

FieldStation42 utilizes several components to recreate the classic television experience.

### `station_42.py`

This command-line tool is used to build catalogs of your video content and generate weekly schedules.  Catalogs are built by inspecting your video content and extracting metadata and need to be rebuilt when your content changes.  Use the UI (`station_42.py`) or command-line arguments (`station_42.py --help`) to manage your catalogs and schedules.

### `field_player.py`

The player application reads the schedule and opens the correct video file, skipping to the appropriate time based on the current schedule. Switching channels starts the correct program at the correct timestamp, simulating continuous broadcast. The player's status is available via `runtime/play_status.socket`.
See [this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for information.

## Connecting Hardware

FieldStation42 is designed to be a software-defined television emulator, so the following are the primary points of integration:

*   **Connecting to a TV:** Uses HDMI output (via a Raspberry Pi or similar hardware).  Vintage TVs will require HDMI-to-Composite or HDMI-to-RF adapters.
*   **Remote Control Integration:**  The player supports external control through `channel.socket` and `play_status.socket`.  See [this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for information.

## Contributing

Contribute to FieldStation42's development:

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request.
4.  Ask questions or join the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## Troubleshooting

*   **Player Startup:** Check video paths and config files.
*   **Video/Audio Issues:** Verify MPV installation and functionality.
*   **Web UI Problems:** Ensure the server is running with `--server` and check your browser's developer tools.
*   **Database Errors:** Confirm file permissions and Python version.
*   **More Help:** Check the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## Alpha Software Disclaimer

This is an alpha-stage project.  Installation requires some experience with the command line, JSON configuration, and video file management.
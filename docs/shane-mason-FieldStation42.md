# FieldStation42: Relive the Golden Age of Television

FieldStation42 is a Python-based TV simulator that lets you create and experience your own custom cable and broadcast TV experience, bringing the nostalgia of vintage television to life.  Explore the project on [GitHub](https://github.com/shane-mason/FieldStation42).

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

## Key Features

*   📺 **Multiple Channels:** Supports numerous simultaneous channels, just like cable!
*   🎬 **Seamless Playback:** Automatically handles commercial breaks and transitions between content.
*   📅 **Automated Scheduling:** Generates weekly schedules based on station configurations.
*   🎬 **Feature-Length Content:** Supports long-form programming, from movies to extended broadcasts.
*   🔄 **Content Freshness:** Randomly selects unwatched shows to keep the lineup interesting.
*   🗓️ **Date-Restricted Content:** Schedule shows for specific seasons or holidays.
*   📡 **Customization:** Offers per-station configurations for sign-off videos and off-air loops.
*   🖥️ **User-Friendly Interface:** Includes a web UI for managing catalogs and schedules.
*   🕹️ **Remote Control Integration:** Optional hardware connections for channel changing.
*   🌐 **Web-Based Remote:** A built-in web-based remote control for easy channel surfing.
*   📺 **On-Screen Display:** Displays channel name, time, date, and customizable icons.
*   📢 **Looping Channels:** Create channels for community bulletins and informational loops.
*   📺 **Preview/Guide Channel:** Feature a guide channel with embedded video and messages.
*   🧩 **Flexible Scheduling:** Supports various channel types: network, commercial-free, and looping.

![A cable box next to a TV](docs/cable_cover_3.png?raw=true)

## Quick Start Guide

Get your retro TV experience up and running with these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place your video files in the appropriate folders within the `catalog/` directory, following the structure defined in the example configurations in `confs/examples/`.
4.  **Configure Your Stations:** Copy an example configuration file from `confs/examples/` to `confs/` and edit it to define your channels.
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

For a detailed, step-by-step guide, refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## Project Structure

*   `station_42.py`: Main CLI and UI for building catalogs and schedules.
*   `field_player.py`: The main TV interface/player.
*   `fs42/`: Core Python modules (catalog, schedule, API, etc.).
*   `confs/`: Station and system configuration files.
*   `catalog/`: Your video content, organized by channel.
*   `runtime/`: Runtime files, sockets, and status.
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS).
*   `docs/`: Images and documentation.

## Installation & Setup

### Detailed Guide

For comprehensive instructions on setting up and administering FieldStation42, see the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## How It Works

FieldStation42 recreates the classic TV experience through its various components.

### `station_42.py`

Use this CLI or UI tool to create catalogs and generate schedules. Catalogs store metadata about station content and need to be rebuilt when content changes, which can take time depending on video quantity. Catalogs and station configuration data build the schedules. Run `station_42.py` without arguments to launch the terminal UI, or use command-line arguments for complete control.  For command-line options, run `station_42.py --help`.

### `field_player.py`

The main TV interface. This player loads the schedule and picks the correct video based on the current time.  Changing channels starts the same process.

The player's status and current channel are written to `runtime/play_status.socket`.  See [this page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for integration information.

### `command_input.py`

This is an example that shows how to connect external programs for channel changes and status information. This script listens on the UART connection and writes commands to `runtime/channel.socket`.

## Connecting to a TV

FieldStation42 supports connecting to vintage TVs.  An HDMI-to-composite or HDMI-to-RF adapter is recommended to connect to older televisions that lack an HDMI port.

## Connecting a Remote Control or Other Device

Integrate your devices and external programs by using the published status and commands through the use of `channel.socket` and `play_status.socket`.  For more information, [see this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script).  For a guide on Bluetooth remotes, [see the discussion boards](https://github.com/shane-mason/FieldStation42/discussions/47).

## Contributing

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request.
4.  For questions, use the [Issues](https://github.com/shane-mason/FieldStation42/issues) or [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## Troubleshooting

*   **Player Won't Start:** Check video file paths and config files.
*   **No Video/Audio:** Ensure MPV is installed and working.
*   **Web UI Issues:** Make sure the server is running with `--server` and check your browser's dev tools.
*   **Database Errors:** Verify file permissions and Python version.

For additional help, consult the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## Important Note: Alpha Software

This project is in early stages of development. Requires some technical skills.

*   Linux command line.
*   Editing JSON config files.
*   Organizing video files.
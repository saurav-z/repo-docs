# FieldStation42: Relive the Golden Age of Television

**FieldStation42 recreates the authentic experience of classic OTA television, complete with scheduled programming, channel surfing, and the comforting glow of a vintage TV.**  [Go to the original repo](https://github.com/shane-mason/FieldStation42).

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

## Key Features

*   **Multiple Channels:** Simulate a full lineup of TV stations.
*   **Realistic Scheduling:**  Automatically generates weekly schedules based on your custom configurations, including commercial breaks and bumps.
*   **Continuous Playback:**  Channels play serially, as if they've been broadcasting the entire time, perfect for channel surfing.
*   **Flexible Content:** Supports movies and shows of any length, with random selection of unwatched content.
*   **Customizable Stations:**  Configure station sign-offs, off-air loops, and date ranges for seasonal content.
*   **Web-Based Remote Control:** Control your simulated TV experience via a web interface.
*   **Looping Channels:** Create channels for community information or continuous content.
*   **Preview/Guide Channel:**  A channel with an embedded video and configurable messages.
*   **Hardware Integration:** Supports optional hardware connections for channel changing.

## Quick Start

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```

2.  **Install Dependencies:**

    ```bash
    ./install.sh
    ```

3.  **Add Your Content:** Place video files in the `catalog/` directory and configure your content in the example files located in the `confs/examples/` folder.
4.  **Configure Stations:**
    *   Copy an example config from `confs/examples` to `confs/` and edit as needed.
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

For a comprehensive guide to setup and administration, please see the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## Project Structure

*   `station_42.py`: Main CLI and UI for building catalogs and schedules
*   `field_player.py`: Main TV interface/player
*   `fs42/`: Core Python modules (catalog, schedule, API, etc.)
*   `confs/`: Station and system configuration files
*   `catalog/`: Your video content, organized by channel
*   `runtime/`: Runtime files, sockets, and status
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS)
*   `docs/`: Images and documentation

## Installation & Setup

For a complete guide to setting up and administering FieldStation42 software, please refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

### Quick Setup Steps

*   **Prerequisites:** Ensure Python 3 and MPV are installed.
*   **Clone the Repository:** `git clone <repository_url>`
*   **Install:** Run the install script (`./install.sh`).
*   **Content:** Add your video content to the appropriate directories.
*   **Configuration:** Configure your stations by copying and editing example configuration files.
*   **Schedule Generation:** Generate a weekly schedule using `python3 station_42.py`.
*   **Launch Player:** Run `field_player.py`.
*   **(Optional) Startup on Boot:**  Use `fs42/hot_start.sh` (for advanced users).

## How FieldStation42 Works

FieldStation42 utilizes several key components to bring you an authentic TV experience.

### station\_42.py

This tool allows you to create and manage content catalogs and generate weekly schedules. Catalogs store metadata about your media content and need to be rebuilt when you update your content. The built schedule takes catalog information and configuration files to build the schedule.  You can manage everything via the command line or a terminal UI. Run `station_42.py --help` to see available options.

### field\_player.py

This is the main TV player interface. It reads the schedule, starts the correct video at the right position based on the current time, and starts a channel. When switching channels, the player keeps track of the playback time to mimic live TV. The player's status is written to `runtime/play_status.socket`.  See [this page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for integration details.

## Connecting to a TV

To connect to a vintage television, you will need an HDMI to composite or RF adapter to convert the output from your Raspberry Pi.

## Connecting a Remote Control or Other Devices

The player supports external commands and publishes its status, so you can connect external devices. See [this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for information on integration and the [discussions](https://github.com/shane-mason/FieldStation42/discussions/47) for details on using a Bluetooth remote.

## Contribute

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request.
4.  Ask questions and join the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## Troubleshooting

*   **Player Won't Start:**  Check video file paths and configuration files.
*   **No Video/Audio:** Ensure MPV is installed and working correctly.
*   **Web UI Issues:** Make sure the server is running with `--server` and check browser's developer tools for errors.
*   **Database Errors:** Verify file permissions and Python version.
*   For more help, consult the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## Links & Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## Important Notes: Alpha Software

This project is under active development, and installation requires some technical experience, including:

*   Basic Linux command line usage.
*   JSON configuration file editing.
*   Movie file conversion and organization.
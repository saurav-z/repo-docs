# FieldStation42: Relive the Golden Age of Television 📺

**FieldStation42 is an open-source TV simulator that brings the nostalgic experience of classic cable and broadcast television to your modern setup.** ([Original Repository](https://github.com/shane-mason/FieldStation42))

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

## Key Features:

*   **Multiple Channels:** Simulate a full channel lineup.
*   **Realistic Programming:** Enjoy shows playing serially with commercial breaks, bumpers, and movie-length content.
*   **Dynamic Schedules:** Automatically generates weekly schedules based on station configurations, with random show selection.
*   **Customization:** Configure station sign-offs, off-air loops, and date ranges for shows (seasonal programming).
*   **User Interface:** Manage catalogs and schedules with the built-in UI.
*   **Web-Based Remote Control:** Control your virtual TV via a web interface.
*   **On-Screen Display:** Channel name, time, and date overlay.
*   **Looping Channels:** Create channels for community bulletins or information loops.
*   **Preview/Guide Channel:** Configurable channel with embedded video and messages.

## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place video files in the appropriate folders as described in the documentation and examples.
4.  **Configure Stations:** Copy and edit example configurations from `confs/examples` into `confs/`.
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

For a full, step-by-step guide, see the [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki).

## 📁 Project Structure

*   `station_42.py`: Main CLI and UI for building catalogs and schedules.
*   `field_player.py`: Main TV interface/player.
*   `fs42/`: Core Python modules (catalog, schedule, API, etc.).
*   `confs/`: Station and system configuration files.
*   `catalog/`: Your video content, organized by channel (created by installer).
*   `runtime/`: Runtime files, sockets, and status (created by installer).
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS).
*   `docs/`: Images and documentation.

## 🛠️ Installation & Setup

A complete setup guide is available in the [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki).

### Quickstart Setup

*   Ensure Python 3 and MPV are installed.
*   Clone the repository.
*   Run the install script.
*   Add your video content.
*   Configure your stations.
*   Generate a weekly schedule.
*   Start the player.
*   Configure start-on-boot (optional).

**Note:** This quickstart is for overview purposes only; detailed instructions are in the [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki).

## How FieldStation42 Works

FieldStation42 uses several components to recreate the old-school TV experience:

### `station_42.py`

Used for building content catalogs and generating schedules. Catalogs store metadata about the content and need to be rebuilt when content changes. The liquid-scheduler builds schedules using catalogs and station configurations. Run `station_42.py --help` for a full list of options.

### `field_player.py`

The primary TV interface. It loads the schedule on startup, starts the correct video file based on the current time, and resumes playback on channel changes. Player status is written to `runtime/play_status.socket`.  Integration details with external programs via `channel.socket` and `play_status.socket` are described in the [wiki](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script).

## Connecting to Your TV

The Raspberry Pi has an HDMI output, but you may need an adapter (HDMI to composite/RF) to connect to older TVs.

## Connecting a Remote Control or Other Device

The player accepts external commands and publishes its status, making it easy to connect external devices. See the [wiki](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for details on integrating with `channel.socket` and `play_status.socket`. A detailed guide on setting up a Bluetooth remote control is available in the [Discussions](https://github.com/shane-mason/FieldStation42/discussions/47).

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make changes and add tests.
3.  Open a pull request describing your changes.
4.  For questions, open an issue or join the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## 🐞 Troubleshooting

*   **Player Won't Start:** Check video paths and config files.
*   **No Video/Audio:** Ensure MPV is installed and working.
*   **Web UI Not Loading:** Run with `--server` and check browser errors.
*   **Database Errors:** Check file permissions and Python version.

For further assistance, consult the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## 📚 Links & Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## Important Notes

This is an alpha project under active development. Requires a basic understanding of Linux, JSON, and video file management.
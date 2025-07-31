# FieldStation42: Recreate the Golden Age of Television 📺

Relive the nostalgia of classic broadcast television with FieldStation42, a software-based TV simulator that brings the authentic feel of OTA TV to your modern setup.  [Check out the project on GitHub](https://github.com/shane-mason/FieldStation42)

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

---

## Key Features

*   **Multiple Channels:** Supports simultaneous channel broadcasting.
*   **Automated Commercials & Bumps:** Seamlessly integrates commercials and channel bumps into content.
*   **Weekly Scheduling:** Generates weekly schedules based on per-station configurations.
*   **Feature-Length Content:** Supports movie-length programming blocks.
*   **Smart Content Selection:** Randomly chooses shows that haven't been played recently to keep content fresh.
*   **Date-Range Support:** Allows scheduling of seasonal content like sports or holiday specials.
*   **Customizable Station Branding:** Includes per-station sign-off videos and off-air loops.
*   **User Interface:** Web interface to manage catalogs and schedules.
*   **Optional Hardware Integration:** Support for hardware channel changing.
*   **Web Remote Control:** Built-in web-based remote control.
*   **On-Screen Display (OSD):** Customizable OSD with channel name, time, date, and icons.
*   **Looping Channels:** Create community bulletin boards and information loops.
*   **Preview/Guide Channel:** Includes embedded video and configurable messages.

---

## 🚀 Quick Start Guide

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place video files in the appropriate folders (`catalog/`).
4.  **Configure Stations:** Copy and edit an example configuration file (`confs/`).
5.  **Build Catalogs & Schedules:**
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

For a detailed walkthrough, refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## 🛠️ Installation & Setup

FieldStation42 requires Python 3, MPV, and some basic Linux command-line experience. Detailed instructions are available in the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## 📁 Project Structure

*   `station_42.py`:  Main CLI and UI for catalog/schedule management.
*   `field_player.py`:  The core TV interface and video player.
*   `fs42/`:  Core Python modules (catalog, schedule, API, etc.).
*   `confs/`:  Station and system configuration files.
*   `catalog/`:  Your video content, organized by channel.
*   `runtime/`:  Runtime files, sockets, and status information.
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS).
*   `docs/`:  Images and documentation.

---

## How It Works

FieldStation42's components work together to simulate a realistic TV experience.

### `station_42.py`

This script is used for catalog creation and schedule generation. Catalogs, which store metadata about video content, are rebuilt whenever the content changes. The liquid-scheduler uses these catalogs and station configurations to build schedules.

### `field_player.py`

This is the main TV interface, playing content based on the current time and channel.  It fetches the current time's scheduled content. Tuning to a previous channel resumes playback from the correct point in time.  Player status is output to `runtime/play_status.socket`.

### `command_input.py`

An example script showing how to connect external hardware and programs to control channel changes. Listens for commands on the pi's UART connection and writes to `runtime/channel.socket`.

---

## 📺 Connecting to a TV

FieldStation42 is designed for use with a Raspberry Pi.  You will need to convert the Pi's HDMI output to a signal your TV understands (Composite, RF).  Use an HDMI to composite or HDMI to RF adapter to convert the signal.

---

## 🔌 Connecting a Remote Control or Other Device

The player accepts external commands and publishes its status via sockets, allowing easy integration with external devices. See [this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for details on integrating with `channel.socket` and `play_status.socket`. For details on using a Bluetooth remote, see the [discussion boards](https://github.com/shane-mason/FieldStation42/discussions/47).

---

## 🤝 Contributing

Contributions are welcome!

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request describing your changes.
4.  Ask questions, or discuss ideas in the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

---

## 🐞 Troubleshooting

*   **Player Won't Start:** Check your video file paths and configuration files.
*   **No Video/Audio:** Ensure MPV is installed and functional.
*   **Web UI Not Loading:** Verify the server is running with `--server` and check your browser's console for errors.
*   **Database Errors:** Check file permissions and Python version.
*   For more assistance, see the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

---

## 📚 Links & Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)
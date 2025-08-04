# FieldStation42: Relive the Golden Age of Television 📺

FieldStation42 is a software-based TV simulator that recreates the authentic experience of watching over-the-air (OTA) television, complete with channel surfing and scheduled programming. Relive the nostalgia with [FieldStation42](https://github.com/shane-mason/FieldStation42).

## Key Features

*   **Simulated Channel Surfing:** Seamlessly switch between channels, with shows continuing as if they've been broadcasting continuously.
*   **Automated Scheduling:** Generates weekly schedules based on your configurations, including commercial breaks and bumpers.
*   **Customizable Content:** Add your own movies, TV shows, and other content to create a personalized viewing experience.
*   **Multiple Channel Support:** Run multiple channels simultaneously.
*   **Flexible Scheduling:** Supports traditional network channels, commercial-free channels, and looping channels.
*   **Web-Based Remote Control:** Integrated web interface for easy channel management.
*   **On-Screen Display:** Displays channel name, time, and date.
*   **Preview/Guide Channel:** With embedded video and configurable messages.

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
3.  **Add Your Content:** Place your video files in the appropriate folders (see `catalog/` and `confs/examples/`).
4.  **Configure Stations:** Copy an example config from `confs/examples` to `confs/` and edit as needed.
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

For a detailed guide, consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## 📁 Project Structure

*   `station_42.py`: Main CLI and UI for building catalogs and schedules.
*   `field_player.py`: Main TV interface/player.
*   `fs42/`: Core Python modules (catalog, schedule, API, etc.).
*   `confs/`: Station and system configuration files.
*   `catalog/`: Your video content, organized by channel.
*   `runtime/`: Runtime files, sockets, and status.
*   `fs42/fs42_server/static/`: Web UI static files (HTML, JS, CSS).
*   `docs/`: Images and documentation.

## 🛠️ Installation & Setup

### Quickstart Setup

1.  Ensure Python 3 and MPV are installed on your system.
2.  Clone the repository.
3.  Run the install script.
4.  Add your own content (videos).
5.  Configure your stations.
    *   Copy an example JSON file from `confs/examples` into `confs/`.
6.  Generate a weekly schedule.
    *   Run `python3 station_42.py` on the command line. Use the `--rebuild_catalog` option if content has changed.
7.  Watch TV.
    *   Run `field_player.py` on the command line.
8.  Configure start-on-boot (optional and not recommended unless you are making a dedicated device).
    *   Run `fs42/hot_start.sh` on the command line.

For comprehensive setup instructions, please refer to the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

## ⚙️ How It Works

FieldStation42 uses multiple components to recreate the feeling of old-school TV:

### `station_42.py`

This is used to create catalogs and generate schedules. Catalogs store metadata, so rebuild them when content changes. Running `station_42.py` with no arguments starts a UI, or you can use command-line arguments for all operations.

### `field_player.py`

This is the main TV interface. It reads the schedule, opens the correct video file, and starts at the correct time. The player writes its status to `runtime/play_status.socket`, which can be monitored. See [this page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for details.

## 🔌 Connecting to a TV

FieldStation42 works with standard HDMI, and can be connected to vintage TVs using HDMI adapters.

## 🕹️ Connecting a Remote Control or Other Device

FieldStation42 supports external control via sockets. See [this wiki page](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script).

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make changes and add tests.
3.  Open a pull request.
4.  For questions, use the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

## 🐞 Troubleshooting

*   **Player won't start:** Check video file paths and config files.
*   **No video/audio:** Ensure MPV is installed.
*   **Web UI not loading:** Run the server with `--server` and check browser dev tools.
*   **Database errors:** Check file permissions and Python version.

For more help, visit the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

## 📚 Links & Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

## Important Notes

This is an alpha-stage project, so installation requires some technical knowledge. Be familiar with the Linux command line, JSON files, and video file organization.
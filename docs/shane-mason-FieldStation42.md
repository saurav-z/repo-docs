# FieldStation42: Relive the Golden Age of Television 📺

**FieldStation42 is a Python-based TV simulator that recreates the authentic experience of watching over-the-air television, complete with channel surfing and scheduled programming.**  ([Original Repository](https://github.com/shane-mason/FieldStation42))

---

## 🔑 Key Features

*   **Multiple Channels:** Simulate several TV channels simultaneously.
*   **Seamless Playback:** Shows continue playing across channel changes as if they had been on the air continuously.
*   **Automated Scheduling:** Generates weekly schedules based on your custom configurations.
*   **Commercial Breaks & Bumps:** Automatically inserts commercial breaks and station bumpers for an authentic feel.
*   **Content Variety:** Supports feature-length content like movies.
*   **Dynamic Programming:** Randomly selects shows that haven't been played recently to keep the lineup fresh.
*   **Date-Restricted Content:** Supports date ranges for seasonal or special event programming.
*   **Custom Station Branding:** Configure station sign-off videos and off-air loops.
*   **Web-Based Control:** Built-in web interface for managing content and schedules.
*   **Optional Hardware Integration:** Supports external hardware for channel changing.
*   **On-Screen Display (OSD):** Displays channel name, time, and date with customizable icons.
*   **Looping Channels:** Ideal for community bulletin boards or information feeds.
*   **Preview Channel:** Guide channel with embedded video and custom messaging.

---

## 🚀 Quick Start Guide

Get started with FieldStation42 in a few simple steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shane-mason/FieldStation42.git
    cd FieldStation42
    ```
2.  **Install Dependencies:**
    ```bash
    ./install.sh
    ```
3.  **Add Your Content:** Place your video files into the `catalog/` directory, organized by channel, and the configuration files.
4.  **Configure Stations:**  Copy an example configuration file from `confs/examples/` to `confs/` and customize it.
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

For detailed instructions, consult the comprehensive [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## 📁 Project Structure

*   `station_42.py`: Main CLI and UI for catalog and schedule management.
*   `field_player.py`:  The primary TV interface/player application.
*   `fs42/`: Core Python modules: catalog, schedule, and API.
*   `confs/`:  Directory for station and system configuration files.
*   `catalog/`: Your video content organized by channel.
*   `runtime/`: Temporary files for running the application (sockets, status).
*   `fs42/fs42_server/static/`:  Web UI static files (HTML, CSS, JavaScript).
*   `docs/`: Documentation and images.

---

## 🛠️ Installation & Setup

Follow the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki) for a complete walkthrough of installation and configuration.

### Quickstart Overview

1.  **Prerequisites:** Ensure you have Python 3 and MPV installed.
2.  **Clone:** Get the repository.
3.  **Run Installer:** Execute the installation script.
4.  **Add Content:**  Populate the catalog with your video files.
5.  **Configure Stations:** Modify the example configuration files.
6.  **Generate Schedule:** Run `python3 station_42.py` to build a schedule.
7.  **Watch TV:** Run `python3 field_player.py` to start the player.

---

## 🕹️ How It Works

FieldStation42 uses multiple components to create an authentic TV experience:

### `station_42.py`

*   **Catalog & Schedule Management:** Creates catalogs to store metadata about the content. Catalogs must be rebuilt when content changes. Uses catalogs and station configurations to generate schedules.  You can manage catalogs and schedules via the terminal UI or command-line arguments (e.g., `station_42.py --help`).

### `field_player.py`

*   **TV Interface:** Reads the schedule and plays the correct video at the correct time, skipping to the proper position for the show. The player's status is written to a socket (`runtime/play_status.socket`) for integration with external programs. See the [wiki](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) for socket information.

## 🔌 Connecting Devices

FieldStation42 allows for easy integration with external devices:

*   **Raspberry Pi:**  Connect to a vintage TV using HDMI-to-composite or HDMI-to-RF adapters.
*   **Remote Control Integration:** Integrate with custom remote controls through the API and socket-based communication.

## 🤝 Contributing

Help improve FieldStation42:

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Submit a pull request.
4.  Ask questions or discuss ideas in the [Discussions](https://github.com/shane-mason/FieldStation42/discussions).

---

## 🐞 Troubleshooting

*   **Player Won't Start:** Verify video file paths and configuration files.
*   **No Video/Audio:** Confirm MPV is installed and operational.
*   **Web UI Problems:** Ensure the server is running (`--server`) and check your browser's developer tools for errors.
*   **Database Issues:**  Verify file permissions and Python version.
*   For additional help, consult the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or create an issue.

---

## 📚 Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

---

## ⚠️ Alpha Software Notice

This is an alpha-stage project.  Installation requires familiarity with the following:

*   Basic Linux command-line usage.
*   Editing JSON configuration files.
*   Movie file management.
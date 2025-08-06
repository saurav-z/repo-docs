# FieldStation42: Relive the Golden Age of Television

Tired of endless streaming options? **FieldStation42** is a cable and broadcast TV simulator that creates an authentic retro TV experience, complete with channel surfing, scheduled programming, and nostalgic charm. Check out the [original repo](https://github.com/shane-mason/FieldStation42)!

![An older TV with an antenna rotator box in the background](docs/retro-tv.png?raw=true)

---

## Key Features

*   **Multiple Channels:** Simulate a full cable or OTA lineup.
*   **Realistic Scheduling:** Plays shows in time slots, just like real TV.
*   **Commercial Breaks & Bumps:** Adds to the authenticity with breaks and station identification.
*   **Content Flexibility:** Supports movies and all video lengths.
*   **Dynamic Content:** Randomly selects unwatched content to keep the lineup fresh.
*   **Configurable:** Station configurations, including sign-off videos and loop channels.
*   **Web UI:** Built-in remote control and content management.
*   **On-Screen Display:** Displays channel information and time/date.
*   **Looping Channels:** Create community bulletin boards or information feeds.
*   **Preview/Guide Channel:** Includes embedded video and configurable messages.
*   **Flexible Scheduling:** Supports various channel types (network, commercial-free, loop).

---

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
4.  **Configure Stations:** Copy and edit example config files in `confs/`.
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

For a full guide, see the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## 📁 Project Structure

*   `station_42.py` — CLI and UI for catalog and schedule management.
*   `field_player.py` — Main TV interface.
*   `fs42/` — Core Python modules.
*   `confs/` — Station and system configuration files.
*   `catalog/` — Video content organized by channel.
*   `runtime/` — Runtime files and status.
*   `fs42/fs42_server/static/` — Web UI static files.
*   `docs/` — Images and documentation.

---

## 🛠️ Installation & Setup - Simplified

### Steps

1.  **Prerequisites:** Ensure you have Python 3 and MPV installed.
2.  **Clone:** Clone the GitHub repository.
3.  **Install:** Run the installation script.
4.  **Content:** Add your video content.
5.  **Configure:** Edit the station configuration files.
6.  **Generate:** Generate a weekly schedule.
7.  **Watch:** Run `field_player.py` to watch TV.

For detailed setup instructions, consult the [FieldStation42 Guide](https://github.com/shane-mason/FieldStation42/wiki).

---

## How It Works

*   **`station_42.py`:** Creates catalogs of your content and schedules. Use the command-line arguments or the terminal UI.
*   **`field_player.py`:** The main TV interface, which starts the correct video and resumes where the previous show left off.

---

## Connecting to a TV

*   **HDMI:** Use a Raspberry Pi with HDMI output.
*   **Vintage TVs:** Use an HDMI to composite or HDMI to RF adapter.

---

## Extending FieldStation42

*   **Remote Control:** You can connect external devices using `channel.socket` and `play_status.socket`. See [Changing Channel From Script](https://github.com/shane-mason/FieldStation42/wiki/Changing-Channel-From-Script) and the [Bluetooth remote guide](https://github.com/shane-mason/FieldStation42/discussions/47).

---

## 🤝 Contributing

1.  Fork the repository and create a feature branch.
2.  Make your changes and add tests.
3.  Open a pull request.
4.  For questions, use the [Discussions](https://github.com/shane-mason/FieldStation42/discussions) or open an issue.

---

## 🐞 Troubleshooting

*   **Player Won't Start:** Check video paths and config files.
*   **No Video/Audio:** Ensure MPV is installed.
*   **Web UI Issues:** Start the server with `--server` and check browser developer tools for errors.

For more help, see the [wiki](https://github.com/shane-mason/FieldStation42/wiki) or open an issue.

---

## 📚 Links & Resources

*   [FieldStation42 Guide (Wiki)](https://github.com/shane-mason/FieldStation42/wiki)
*   [API Reference](fs42/fs42_server/README.md)
*   [Discussions](https://github.com/shane-mason/FieldStation42/discussions)
*   [Releases](https://github.com/shane-mason/FieldStation42/releases)
*   [Issues](https://github.com/shane-mason/FieldStation42/issues)

---

## Important Notes

*   This is an alpha project in active development.
*   Requires basic Linux command-line knowledge and editing JSON config files.
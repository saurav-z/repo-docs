# Automatic Ripping Machine (ARM): Automate Your Disc Ripping!

ARM simplifies the process of ripping Blu-rays, DVDs, and CDs, providing a fully automated solution for your media library.  [Visit the original repository](https://github.com/automatic-ripping-machine/automatic-ripping-machine) for more details and contributions.

## Key Features

*   **Automated Disc Detection:** Detects disc insertion via `udev`.
*   **Intelligent Disc Type Handling:** Automatically identifies disc types (video, audio, data).
*   **Video Ripping (Blu-ray/DVD):**
    *   Retrieves title and year from disc or [OMDb API](http://www.omdbapi.com/).
    *   Determines if video is Movie or TV using [OMDb API](http://www.omdbapi.com/).
    *   Rips using MakeMKV or HandBrake (all features or main feature).
    *   Ejects disc and queues Handbrake transcoding asynchronously.
    *   Integrates with Plex/Emby for easy library management.
    *   Sends notifications via IFTTT, Pushbullet, Slack, Discord, and more.
*   **Audio Ripping (CD):** Rips CDs using `abcde`, retrieving disc data and album art from [musicbrainz](https://musicbrainz.org/).
*   **Data Backup (Blu-ray/DVD/CD):** Creates ISO backups of data discs.
*   **Headless Operation:** Designed for server environments.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web UI:** Provides a Python Flask UI for managing jobs, viewing logs, and updating settings.

## How it Works

1.  Insert a disc (Blu-ray, DVD, or CD).
2.  ARM automatically detects the disc and determines its type.
3.  ARM rips the disc based on its type (video, audio, or data).
4.  The disc is ejected, and any further processing (like transcoding) is handled automatically.

## System Requirements

*   A system capable of running Docker containers.
*   One or more optical drives for ripping Blu-rays, DVDs, and CDs.
*   Sufficient storage space (a NAS is recommended) to store your ripped media.

## Installation

*   **Docker:**  See the [Docker Installation Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker) for instructions.
*   **General Installation:**  See the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/) for comprehensive installation instructions.

## Troubleshooting

*   Refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/) for troubleshooting common issues.

## Contributing

Contributions are welcome! See the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide) for details on how to contribute.  Consider submitting a how-to to the wiki if you set up ARM in a different environment.

## License

[MIT License](LICENSE)
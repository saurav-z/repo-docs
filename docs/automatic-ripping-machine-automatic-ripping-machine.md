# Automatic Ripping Machine (ARM): Automate Your Media Ripping and Backup

ARM is a powerful, open-source tool designed to automatically rip and back up your Blu-ray, DVD, and CD media, making your media library more accessible.

[Original Repository](https://github.com/automatic-ripping-machine/automatic-ripping-machine)

## Key Features

*   **Automated Disc Detection:** Automatically detects disc insertion via `udev`.
*   **Intelligent Disc Type Identification:** Determines disc type (video, audio, or data).
*   **Video Ripping (Blu-ray & DVD):**
    *   Retrieves movie titles from disc or [OMDb API](http://www.omdbapi.com/) for organized storage.
    *   Identifies movie or TV show using [OMDb API](http://www.omdbapi.com/).
    *   Rips using MakeMKV or HandBrake (all features or main feature).
    *   Asynchronously batches transcoding jobs for efficient processing.
    *   Integrates with notification services (IFTTT, Pushbullet, Slack, Discord, etc.).
*   **Audio Ripping (CD):** Rips CDs using `abcde`, retrieves disc data and album art from [musicbrainz](https://musicbrainz.org/).
*   **Data Backup:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, and CD).
*   **Headless Operation:** Designed for server environments, minimizing user interaction.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web UI:** Provides a Python Flask UI for managing jobs, viewing logs, and more.

## How it Works

1.  Insert a disc into your optical drive.
2.  ARM automatically identifies the disc type and starts the ripping process.
3.  Once complete, the disc is ejected, and any further processing steps are queued.

## System Requirements

*   A system capable of running Docker containers (recommended).
*   One or more optical drives (Blu-ray/DVD/CD readers).
*   Ample storage space (NAS recommended) for your media library.

## Installation

*   For a standard installation, refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   For Docker installation instructions, see the [wiki Docker guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   Find solutions to common issues in the [wiki troubleshooting guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

*   Contributions are welcome. See the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide) for details.
*   Share your setup experiences by submitting a how-to guide to the wiki.

## License

*   [MIT License](LICENSE)
# Automatic Ripping Machine (ARM): Automate Your Disc Ripping

**Automatic Ripping Machine (ARM) simplifies the process of ripping your Blu-rays, DVDs, and CDs, offering a fully automated solution for your media library.** Check out the original repo [here](https://github.com/automatic-ripping-machine/automatic-ripping-machine).

## Key Features

*   **Automated Disc Detection:** Automatically detects disc insertion using udev.
*   **Intelligent Disc Type Identification:** Identifies disc type (video, audio, or data) and processes accordingly.
*   **Video Ripping:**
    *   Retrieves movie titles and details from disc or the OMDb API.
    *   Determines if video is Movie or TV using OMDb API.
    *   Rips using MakeMKV or HandBrake (all features or main feature).
    *   Ejects disc and queues up Handbrake transcoding.
    *   Asynchronous transcoding job batching.
    *   Integrates with Plex or Emby for easy library management.
    *   Sends notifications via IFTTT, Pushbullet, Slack, Discord, and more.
*   **Audio Ripping:** Rips CDs using abcde, fetching disc data and album art from musicbrainz.
*   **Data Backup:** Creates ISO backups for data discs.
*   **Headless Operation:** Designed to run on a server without a graphical interface.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web Interface:** Python Flask UI to manage jobs, view logs, and update settings.

## Getting Started

1.  **Insert a disc:** Place your Blu-ray, DVD, or CD into the optical drive.
2.  **Automatic Processing:** ARM will automatically identify the disc and begin the ripping process.
3.  **Wait for Completion:** The disc will be ejected once ripping and transcoding are complete (if applicable).
4.  **Enjoy your media!** The ripped files will be available for viewing.

## Requirements

*   A system capable of running Docker containers.
*   One or more optical drives (Blu-ray, DVD, CD).
*   Sufficient storage space (NAS recommended).

## Installation

Refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/) for detailed installation instructions.

## Troubleshooting

For troubleshooting tips and solutions, please consult the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

Contributions are welcome! Please refer to the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).

## License

[MIT License](LICENSE)
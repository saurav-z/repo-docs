# Automatic Ripping Machine (ARM): Automate Your Disc Ripping!

**Tired of manually ripping your Blu-rays, DVDs, and CDs?**  Automatic Ripping Machine (ARM) is a powerful and automated solution designed to streamline your media ripping process.  Visit the original repository for more information: [https://github.com/automatic-ripping-machine/automatic-ripping-machine](https://github.com/automatic-ripping-machine/automatic-ripping-machine)

## Key Features

*   **Automatic Disc Detection:** Detects disc insertion using udev, automatically identifying disc types (Blu-ray, DVD, CD).
*   **Intelligent Ripping:**
    *   **Video Ripping (Blu-ray/DVD):** Retrieves titles, determines movie/TV show status (using OMDb API), rips using MakeMKV or HandBrake (including main feature selection), and automatically queues transcoding.
    *   **Audio Ripping (CD):** Rips CDs using abcde, retrieves disc data, and fetches album art from MusicBrainz.
    *   **Data Backup:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, CD).
*   **Headless Operation:** Designed to run seamlessly on a server, ideal for automated operation.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously for faster processing.
*   **Web UI:** Includes a Python Flask UI to monitor jobs, view logs, and manage your ripping tasks.
*   **Notification Support:** Sends notifications via various services (IFTTT, Pushbullet, Slack, Discord, and more).

## How to Use ARM

1.  Insert your disc (Blu-ray, DVD, or CD) into the drive.
2.  ARM will automatically detect and begin ripping.
3.  Wait for the disc to eject, indicating the process is complete.
4.  Your ripped media will be ready for viewing!

## System Requirements

*   A system capable of running Docker containers (recommended).
*   One or more optical drives for Blu-ray, DVD, and CD ripping.
*   Sufficient storage space (consider a NAS) for your ripped media.

## Installation

*   **Normal Installation:** [See the Wiki for detailed instructions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   **Docker Installation:** [See the Wiki for Docker-specific instructions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   [Refer to the Wiki for troubleshooting tips and solutions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

*   Pull requests are welcome!  See the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).
*   Share your ARM setup and configurations by contributing to the wiki.

## License

*   [MIT License](LICENSE)
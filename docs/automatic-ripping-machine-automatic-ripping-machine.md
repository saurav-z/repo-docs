# Automatic Ripping Machine (ARM): Automate Your Disc Ripping with Ease

Tired of manually ripping Blu-rays, DVDs, and CDs? **Automatic Ripping Machine (ARM) is a powerful, automated solution for ripping your optical media, making your digital library effortless to build and maintain.**  [Check out the original repo for more details](https://github.com/automatic-ripping-machine/automatic-ripping-machine).

## Key Features

*   **Automated Disc Detection:** Automatically detects the insertion of discs (Blu-ray, DVD, CD) using udev.
*   **Intelligent Disc Type Recognition:**  Identifies disc types (Video, Audio, Data) and handles each appropriately.
    *   **Video Ripping (Blu-ray & DVD):**
        *   Retrieves title information from the disc or the [OMDb API](http://www.omdbapi.com/) for accurate naming ("Movie Title (Year)") for easy organization.
        *   Determines if video is a Movie or TV show using [OMDb API](http://www.omdbapi.com/).
        *   Rips using MakeMKV or HandBrake (supports ripping all features or the main feature).
        *   Automated ejection and Handbrake transcoding queueing.
        *   Asynchronous batch processing of transcoding jobs.
        *   Notification support via various services (IFTTT, Pushbullet, Slack, Discord, and more!).
    *   **Audio Ripping (CD):** Rips audio CDs using `abcde`, including album art and metadata from [musicbrainz](https://musicbrainz.org/).
    *   **Data Disc Handling:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, or CD).
*   **Headless Operation:** Designed to run seamlessly on a server.
*   **Parallel Ripping:** Supports ripping from multiple optical drives concurrently.
*   **Web-Based User Interface:**  Python Flask UI for monitoring jobs, viewing logs, and managing your ripping process.

## Getting Started

1.  **Insert the disc.**
2.  **Wait for the disc to eject automatically.**
3.  **Enjoy your ripped content!**

## System Requirements

*   A system capable of running Docker containers.
*   One or more optical drives (Blu-ray/DVD/CD).
*   Sufficient storage space (NAS recommended).

## Installation

*   [For standard installation, please consult the wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   [For Docker installation instructions, see here](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   [Refer to the wiki for troubleshooting steps](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

Contributions are welcome! Please see the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide) for more information.  Consider submitting how-tos to the wiki for different environments.

## License

This project is licensed under the [MIT License](LICENSE).
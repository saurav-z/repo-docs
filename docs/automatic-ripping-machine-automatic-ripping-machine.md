# Automatic Ripping Machine (ARM): Automate Your Digital Media Library

**Effortlessly rip your Blu-rays, DVDs, and CDs into a digital format with the Automatic Ripping Machine (ARM).**  This open-source project simplifies the process of digitizing your physical media collection.

[Visit the Original Repository](https://github.com/automatic-ripping-machine/automatic-ripping-machine)

## Key Features

*   **Automated Disc Detection:** Automatically detects when a disc (Blu-ray, DVD, or CD) is inserted.
*   **Intelligent Disc Type Recognition:** Determines the disc type (video, audio, or data).
*   **Video Ripping:**
    *   Retrieves movie titles and other metadata.
    *   Rips using MakeMKV or HandBrake.
    *   Handles both movies and TV shows.
    *   Automatically names folders for easy organization with Plex, Emby, etc.
    *   Supports concurrent transcoding jobs.
*   **Audio Ripping:** Rips CDs using `abcde`, retrieving disc data and album art from [MusicBrainz](https://musicbrainz.org/).
*   **Data Backup:** Creates ISO backups for data discs.
*   **Headless Operation:** Designed to be run on a server without a graphical interface.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web Interface:**  A Python Flask UI to manage jobs, view logs, and update settings.
*   **Notifications:**  Send notifications via IFTTT, Pushbullet, Slack, Discord, and more.

## Getting Started

1.  **Insert Disc:** Simply insert your Blu-ray, DVD, or CD into the drive.
2.  **Automated Process:** ARM will automatically identify and begin the ripping process.
3.  **Disc Ejection:** Once complete, the disc will be ejected.
4.  **Repeat:** Repeat the process for each disc in your collection.

## Requirements

*   A system capable of running Docker containers.
*   One or more optical drives (Blu-ray, DVD, CD).
*   Sufficient storage space (NAS recommended) for your digital media library.

## Installation

*   **General Installation:** [See the Wiki for installation instructions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   **Docker Installation:** [See the Wiki for Docker installation instructions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   [Refer to the Wiki for troubleshooting guidance](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

We welcome contributions from the community!  Please review the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).  Consider sharing your ARM setup configurations in the wiki.

## License

[MIT License](LICENSE)
# Automatic Ripping Machine (ARM): Automate Your Media Ripping

**Effortlessly rip and archive your Blu-rays, DVDs, and CDs with the Automatic Ripping Machine (ARM).**

**[Visit the original repository on GitHub](https://github.com/automatic-ripping-machine/automatic-ripping-machine)**

## Key Features

*   **Automated Disc Detection & Ripping:** Automatically detects inserted discs (Blu-ray, DVD, CD) and initiates the ripping process.
*   **Intelligent Disc Type Recognition:** Determines the disc type (video, audio, or data) and processes it accordingly.
    *   **Video (Blu-ray/DVD):**
        *   Retrieves titles from the disc or the OMDb API for accurate naming ("Movie Title (Year)")
        *   Identifies movies or TV shows using the OMDb API.
        *   Rips using MakeMKV or HandBrake (supports full or main feature rips).
        *   Asynchronously queues Handbrake transcoding after ripping.
        *   Sends notifications via IFTTT, Pushbullet, Slack, Discord, and more.
    *   **Audio (CD):** Rips CDs using abcde and retrieves disc data and album art from MusicBrainz.
    *   **Data:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, or CD).
*   **Headless Operation:** Designed to run seamlessly on a server without user interaction.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously for increased efficiency.
*   **Web UI:** A Python Flask UI allows you to manage ripping jobs, view logs, and update settings.

## How to Use ARM

1.  Insert the disc.
2.  Wait for the disc to eject.
3.  Repeat for your entire media collection!

## Requirements

*   A system capable of running Docker containers.
*   One or more optical drives for Blu-rays, DVDs, and CDs.
*   Ample storage space (a NAS is recommended) to store ripped media.

## Installation

*   [Normal Installation Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/)
*   [Docker Installation Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker)

## Troubleshooting

*   [Troubleshooting Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/)

## Contributing

We welcome contributions! Please see the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).

## License

[MIT License](LICENSE)
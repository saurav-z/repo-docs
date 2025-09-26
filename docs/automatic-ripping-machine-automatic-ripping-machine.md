# Automatic Ripping Machine (ARM): Automate Your Disc Ripping

**Automatic Ripping Machine (ARM) is a powerful, open-source tool that automates the process of ripping Blu-ray, DVD, and CD media to digital formats.**

**[View the original repository on GitHub](https://github.com/automatic-ripping-machine/automatic-ripping-machine)**

## Key Features

*   **Automated Disc Detection:** Automatically detects inserted discs using `udev`.
*   **Intelligent Disc Type Identification:** Determines disc type (video, audio, or data).
*   **Video Ripping:**
    *   Retrieves movie titles from the disc or the OMDb API.
    *   Identifies movies or TV shows using the OMDb API.
    *   Rips using MakeMKV or HandBrake (all features or main feature).
    *   Ejects discs and queues Handbrake transcoding after ripping.
    *   Asynchronously batches transcoding jobs.
    *   Sends notifications via IFTTT, Pushbullet, Slack, Discord, and more.
*   **Audio Ripping:** Rips CDs using `abcde`, retrieving disc data and album art from `musicbrainz.org`.
*   **Data Backup:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, CD).
*   **Headless Operation:** Designed for server environments.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web Interface:** Includes a Python Flask UI to manage jobs, view logs, and update settings.

## Getting Started

1.  **Insert a Disc:** Place your Blu-ray, DVD, or CD into the optical drive.
2.  **Automatic Processing:** ARM will automatically detect, rip, and process the disc.
3.  **Eject and Repeat:** The disc will be ejected when processing is complete.

## System Requirements

*   A system capable of running Docker containers is **recommended**
*   One or more optical drives for ripping Blu-rays, DVDs, and CDs.
*   Adequate storage space (consider a NAS) for your ripped media.

## Installation

*   For detailed installation instructions, please refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   For Docker installation instructions, please refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   For troubleshooting assistance, please consult the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

*   Contributions are welcome! See the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).
*   Consider submitting a how-to to the wiki if you set up ARM in a unique environment.

## License

*   [MIT License](LICENSE)
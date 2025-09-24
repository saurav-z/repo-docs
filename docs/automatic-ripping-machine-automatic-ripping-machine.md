# Automatic Ripping Machine (ARM): Automate Your Media Library

ARM empowers you to effortlessly digitize your physical media collection, transforming Blu-rays, DVDs, and CDs into a convenient digital library.  [Visit the original repository](https://github.com/automatic-ripping-machine/automatic-ripping-machine) for detailed information and source code.

## Key Features

*   **Automated Disc Detection:** Automatically recognizes inserted discs using `udev`.
*   **Intelligent Disc Type Identification:** Determines disc type: video (Blu-ray, DVD), audio (CD), or data.
*   **Smart Video Ripping:**
    *   Retrieves movie titles and information from the disc or the OMDb API to name folders correctly for Plex/Emby.
    *   Identifies movies or TV shows using the OMDb API.
    *   Rips using MakeMKV or HandBrake (supports all features or the main feature).
    *   Ejects discs and queues Handbrake transcoding asynchronously.
    *   Sends notifications through various services: IFTTT, Pushbullet, Slack, Discord, and more!
*   **CD Ripping:** Rips audio CDs with album art and metadata from MusicBrainz using `abcde`.
*   **Data Disc Backup:** Creates ISO backups of Blu-ray, DVD, DVD-Audio, or CD data discs.
*   **Headless Operation:** Designed for server environments, allowing for unattended operation.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously.
*   **Web Interface:** Python Flask UI to manage ripping jobs, view logs, and update settings.

## Getting Started

1.  **Insert a disc:** Place your Blu-ray, DVD, or CD into the drive.
2.  **Automatic Ripping:** ARM will automatically detect the disc and begin ripping.
3.  **Enjoy:**  The disc will be ejected, and the digital content will be available in your designated directory.

## System Requirements

*   A system capable of running Docker containers is recommended for ease of setup.
*   One or more optical drives (Blu-ray, DVD, CD) are required for ripping.
*   Sufficient storage space (NAS recommended) to store ripped media files.

## Installation

*   **Detailed Installation Guides:** Refer to the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/) for comprehensive installation instructions.  This includes information on:
    *   [Standard installation](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/)
    *   [Docker Installation](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker)

## Troubleshooting & Support

*   **Troubleshooting:** Consult the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/) for troubleshooting tips and common issues.
*   **Get Help:**  Join the [Discord](https://discord.gg/FUSrn8jUcR) for support and community interaction.

## Contributing

*   **Contributions Welcome:**  We encourage contributions! Submit pull requests.
*   **Documentation:** Consider creating a "howto" guide in the [wiki](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki) if you've successfully set up ARM in a unique environment.
*   **Contributing Guide:** See the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
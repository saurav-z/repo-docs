# Automatic Ripping Machine (ARM): Automate Your Disc Ripping

**Effortlessly rip your Blu-rays, DVDs, and CDs to a digital format with the Automatic Ripping Machine (ARM).**

[![CI](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/main.yml/badge.svg)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/main.yml)
[![Publish Docker Image](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/publish-image.yml/badge.svg)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/publish-image.yml)
[![Docker](https://img.shields.io/docker/pulls/automaticrippingmachine/automatic-ripping-machine.svg)](https://hub.docker.com/r/automaticrippingmachine/automatic-ripping-machine)
[![GitHub forks](https://img.shields.io/github/forks/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/network)
[![GitHub stars](https://img.shields.io/github/stars/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/pulls)
[![GitHub contributors](https://img.shields.io/github/contributors/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/automatic-ripping-machine/automatic-ripping-machine?)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/commits/main)
[![GitHub license](https://img.shields.io/github/license/automatic-ripping-machine/automatic-ripping-machine)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/blob/main/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/automatic-ripping-machine/automatic-ripping-machine?label=Latest%20Stable%20Version)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/releases)
[![GitHub release Date](https://img.shields.io/github/release-date/automatic-ripping-machine/automatic-ripping-machine?label=Latest%20Stable%20Released)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/releases)
![Python Versions](https://img.shields.io/badge/Python_Versions-3.9_|_3.10_|_3.11_|_3.12-blue?logo=python)
[![Wiki](https://img.shields.io/badge/Wiki-Get%20Help-brightgreen)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki)
[![Discord](https://img.shields.io/discord/576479573886107699)](https://discord.gg/FUSrn8jUcR)

## Key Features

*   **Automated Disc Detection:** Automatically identifies inserted discs (Blu-ray, DVD, CD).
*   **Intelligent Disc Type Handling:**
    *   **Video (Blu-ray/DVD):**
        *   Retrieves movie/TV titles from disc or [OMDb API](http://www.omdbapi.com/) for optimal naming (e.g., "Movie Title (Year)").
        *   Determines movie or TV show using [OMDb API](http://www.omdbapi.com/).
        *   Rips using MakeMKV or HandBrake (full or main feature).
        *   Ejects disc and queues Handbrake transcoding.
        *   Asynchronously batches transcoding jobs.
        *   Supports notifications via IFTTT, Pushbullet, Slack, Discord, and more.
    *   **Audio (CD):** Rips using abcde, retrieving disc data and album art from [musicbrainz](https://musicbrainz.org/).
    *   **Data (Blu-ray, DVD, DVD-Audio, CD):** Creates ISO backup.
*   **Headless Operation:** Designed for server environments.
*   **Parallel Ripping:** Rips from multiple optical drives simultaneously.
*   **Web UI:** Python Flask UI for job management, log viewing, and updates.

## Getting Started

1.  **Insert Disc:** Simply insert your Blu-ray, DVD, or CD.
2.  **Automated Process:** ARM will automatically detect, rip, and process the disc.
3.  **Eject Disc:** Wait for the disc to eject after processing.
4.  **Repeat:** Insert the next disc to rip.

## Requirements

*   A system capable of running Docker containers (recommended).
*   One or more optical drives.
*   Ample storage space (a NAS is highly recommended).

## Installation

*   **Normal Installation:** [See the wiki for detailed instructions.](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/)
*   **Docker Installation:** [Find Docker-specific instructions here.](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker)

## Troubleshooting

*   [Refer to the wiki for troubleshooting guidance.](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/)

## Contributing

Contributions are welcome! Please review the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide) for information on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).

**For more information and the latest updates, visit the original repository: [Automatic Ripping Machine GitHub](https://github.com/automatic-ripping-machine/automatic-ripping-machine).**
# Automatic Ripping Machine (ARM): Automate Your Disc Ripping with Ease

**ARM simplifies the process of ripping your Blu-ray, DVD, and CD collections, making digital backups effortless.** (Original Repository: [https://github.com/automatic-ripping-machine/automatic-ripping-machine](https://github.com/automatic-ripping-machine/automatic-ripping-machine))

[![CI](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/main.yml/badge.svg)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/main.yml) [![Publish Docker Image](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/publish-image.yml/badge.svg)](https://github.com/automatic-ripping-machine/automatic-ripping-machine/actions/workflows/publish-image.yml)
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

*   **Automated Disc Detection:** Automatically detects disc insertion using udev.
*   **Intelligent Disc Type Recognition:** Determines the disc type (Blu-ray, DVD, CD, data) and initiates appropriate ripping process.
*   **Movie & TV Show Metadata:** Retrieves movie and TV show titles from the disc or [OMDb API](http://www.omdbapi.com/) for organized library naming.
*   **Flexible Ripping Options:** Rips Blu-rays and DVDs using MakeMKV or HandBrake, offering choices for all features or main feature extraction.
*   **Asynchronous Transcoding:** Queues Handbrake transcoding jobs after ripping for efficient processing.
*   **CD Ripping:** Rips audio CDs using abcde, retrieving disc data and album art from [musicbrainz](https://musicbrainz.org/).
*   **ISO Backup:** Creates ISO backups for data discs (Blu-ray, DVD, DVD-Audio, CD).
*   **Headless Operation:** Designed for server environments, running without a graphical interface.
*   **Parallel Ripping:** Supports ripping from multiple optical drives simultaneously for faster processing.
*   **Web UI:** Provides a Python Flask UI for managing ripping jobs, viewing logs, and updating settings.
*   **Notification Support:** Sends notifications via IFTTT, Pushbullet, Slack, Discord, and more.

## How to Use

1.  Insert your disc (Blu-ray, DVD, or CD).
2.  Wait for the disc to automatically eject after ripping.
3.  Repeat for your entire collection!

## Requirements

*   A system capable of running Docker containers.
*   One or more optical drives to rip Blu-rays, DVDs, and CDs.
*   Sufficient storage space (NAS recommended) for your digital media library.

## Installation

*   **General Installation:** [Please see the wiki for installation instructions](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).
*   **Docker Installation:** [Instructions available here](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/docker).

## Troubleshooting

*   [Refer to the wiki for troubleshooting guidance](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/).

## Contributing

Contributions are welcome! Please review the [Contributing Guide](https://github.com/automatic-ripping-machine/automatic-ripping-machine/wiki/Contributing-Guide).  Consider submitting a "howto" guide to the wiki if you have set up ARM in a unique environment.

## License

[MIT License](LICENSE)
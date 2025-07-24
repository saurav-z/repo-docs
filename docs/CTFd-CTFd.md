# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and easy-to-use open-source platform designed for creating and managing Capture The Flag (CTF) competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd

CTFd offers a comprehensive suite of features to make setting up and running CTFs a breeze:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges and hints.
    *   Offers a flexible challenge plugin architecture for custom challenge types.
    *   Includes support for static and regex-based flags.
    *   Allows file uploads to the server or an S3-compatible backend.
    *   Provides options for limiting challenge attempts and hiding challenges.
    *   Custom flag plugins
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Reporting:**
    *   Features a scoreboard with automatic tie resolution.
    *   Offers options to hide scores and freeze them at specific times.
    *   Includes scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management & Communication:**
    *   Provides a Markdown content management system.
    *   Includes SMTP and Mailgun email support with confirmation and password reset functionality.
    *   Supports automatic competition starting and ending.
*   **Team Management:**
    *   Allows for team management, hiding, and banning.
*   **Customization:**
    *   Highly customizable through plugin and theme interfaces.
*   **Data Management:**
    *   Offers importing and exporting of CTF data for archival.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` to start the server in debug mode.

### Docker

*   **Using pre-built Docker images:**
    ```bash
    docker run -p 8000:8000 -it ctfd/ctfd
    ```

*   **Using Docker Compose:**
    ```bash
    docker compose up
    ```

For detailed deployment instructions and other options, please see the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd here: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and engage with the community through the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special project needs, feel free to [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, check out the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on functionality. Registering your CTF event with MLC enables automatic user login, score tracking, writeup submission, and event notifications.

To integrate with MajorLeagueCyber, register an account, create an event, and configure the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
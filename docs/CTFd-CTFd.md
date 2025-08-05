# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, open-source Capture The Flag (CTF) platform designed to make running your own cybersecurity competitions easy and customizable.**  ([View on GitHub](https://github.com/CTFd/CTFd))

## Key Features

CTFd offers a comprehensive set of features to create engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Support for dynamic scoring, unlockable challenges, and custom challenge types through a plugin architecture.
    *   Support for static & regex-based flags and custom flag plugins
    *   File uploads to the server or an Amazon S3-compatible backend
    *   Ability to limit challenge attempts and hide challenges.
*   **Competition Modes:**
    *   Individual and team-based competitions.
*   **Scoreboard & Scoring:**
    *   Automated tie resolution.
    *   Option to hide scores and freeze the scoreboard at specific times.
    *   Scoregraphs and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system for challenge descriptions and announcements.
    *   SMTP and Mailgun email support, including confirmation and password reset features.
*   **Event Control:**
    *   Automatic competition starting and ending times.
    *   Team management, hiding, and banning capabilities.
*   **Customization:**
    *   Highly customizable through plugin and theme interfaces.
    *   Import and export CTF data.
*   **Bruteforce protection**

## Installation

Follow these steps to get CTFd up and running:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependency installation via apt.
2.  **Configure:** Modify `CTFd/config.ini` to match your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start the server in debug mode.

**Docker:**

*   **Run using Docker Image:**  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run using Docker Compose:**  `docker compose up`  (from the source repository)

For detailed deployment instructions and a getting started guide, please refer to the official [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Explore a live demonstration of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and connect with the community:

*   **Community Forum:** [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:** For commercial support or special projects, contact us via [CTFd contact form](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, visit the official [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on. Register your CTF event to enable features such as automatic login, score tracking, writeup submission, and event notifications.

To integrate with MajorLeagueCyber:
1.  Register an account.
2.  Create an event.
3.  Install the client ID and client secret in the relevant portion in `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
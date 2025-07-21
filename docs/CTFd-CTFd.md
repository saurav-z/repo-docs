# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/mysql.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/lint.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and easy-to-use open-source framework for hosting Capture The Flag (CTF) competitions, perfect for cybersecurity enthusiasts and educators.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd provides a comprehensive platform for creating, managing, and running CTF events, including:

*   **Challenge Management:**
    *   Create and customize challenges, categories, hints, and flags directly from the admin interface.
    *   Supports dynamic scoring, unlockable challenges, and a plugin architecture for custom challenges.
    *   Includes support for static and regex-based flags, and custom flag plugins.
    *   Offers file uploads to the server or Amazon S3-compatible backends.
    *   Allows limiting challenge attempts and hiding challenges.
    *   Includes unlockable hints.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Reporting:**
    *   Automatic tie resolution on the scoreboard.
    *   Option to hide scores from the public and freeze scores at a specific time.
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Integrated Markdown content management system.
    *   SMTP and Mailgun email support with email confirmation and password recovery features.
*   **Event Management:**
    *   Automatic competition start and end times.
    *   Team management features, including hiding and banning.
*   **Customization:**
    *   Extensive customization options via [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Management:**
    *   Import and export CTF data for archival and sharing.
*   **And much more!**

## Getting Started

To get started with CTFd:

1.  **Install Dependencies:** Run `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies via apt.
2.  **Configure:** Modify `CTFd/config.ini` to match your needs.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker Usage:**

*   **Run using Docker:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run using Docker Compose:** `docker compose up` (from the source repository)

For detailed deployment instructions and a getting started guide, refer to the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Experience CTFd firsthand at the live demo: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us via [https://ctfd.io/contact/](https://ctfd.io/contact/) for commercial support.

## Managed Hosting

For those seeking a hassle-free CTFd experience, explore [managed CTFd deployments](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on capabilities. Enhance your CTF experience by:

1.  Registering an account and creating an event on MajorLeagueCyber.
2.  Installing the client ID and secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
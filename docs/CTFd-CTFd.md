# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly and highly customizable open-source platform for hosting and managing Capture The Flag (CTF) competitions.** ([View the source on GitHub](https://github.com/CTFd/CTFd))

## Key Features

CTFd provides a comprehensive feature set to create engaging and dynamic CTF experiences:

*   **Challenge Creation & Management:**
    *   Admin interface to create challenges, categories, hints, and flags.
    *   Support for dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Challenge plugin architecture for custom challenge types.
    *   Static & Regex-based flags, and custom flag plugins.
    *   Unlockable hints.
    *   File uploads to server or S3-compatible backend.
    *   Challenge attempt limits and challenge hiding.
*   **Competition Modes:**
    *   Individual and team-based competitions.
    *   Users can play individually or form teams.
*   **Scoring & Leaderboards:**
    *   Scoreboard with automatic tie resolution.
    *   Options to hide scores.
    *   Score freezing at a specific time.
    *   Scoregraphs for the top 10 teams and team progress.
*   **Content & Communication:**
    *   Markdown content management system for rich content.
    *   SMTP and Mailgun email support.
    *   Email confirmation and password reset support.
*   **Competition Control:**
    *   Automatic competition starting and ending.
    *   Team management tools (hiding and banning).
*   **Customization & Integration:**
    *   Highly customizable with [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Importing and exporting of CTF data.

## Installation

Get started with CTFd in a few steps:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to your specifications.
3.  **Run:** Use `python serve.py` or `flask run` to run in debug mode.

**Docker:**  Run CTFd quickly using Docker:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or use Docker Compose:

```bash
docker compose up
```

For detailed installation and deployment options, refer to the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Explore CTFd in action:  [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the community:

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** Contact us for commercial support or custom projects:  [https://ctfd.io/contact/](https://ctfd.io/contact/)

## Managed Hosting

For managed CTFd deployments without managing infrastructure, visit [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is seamlessly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker that provides event scheduling, team tracking, and single sign-on. Integrate CTFd with MajorLeagueCyber for automatic user login, score tracking, writeup submission, and event notifications.

1.  Create an account and event on MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
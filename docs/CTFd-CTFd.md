# CTFd: The Open-Source Capture The Flag Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly and customizable open-source platform, ideal for hosting engaging Capture The Flag (CTF) competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

*   **Easy Challenge Creation & Management:**
    *   Admin interface for creating challenges, categories, hints, and flags.
    *   Support for dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Plugin architecture for custom challenges.
    *   Static & Regex based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads to the server or Amazon S3-compatible backends.
    *   Challenge attempt limits and challenge hiding.
    *   Automatic brute-force protection.
*   **Competition Modes:**
    *   Individual and team-based competitions.
*   **Scoreboard & Reporting:**
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing at a specific time.
    *   Score graphs comparing team progress.
*   **Content Management & Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support.
    *   Email confirmation and password reset support.
    *   Automatic competition start and end.
*   **Team & User Management:**
    *   Team management, hiding, and banning.
*   **Customization & Integration:**
    *   Extensive customization via [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Import and export CTF data.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies (apt).
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

**Docker:**

*   Run a container: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose: `docker compose up` (from the source repo)

**For more details on deployment, check out the [CTFd Docs](https://docs.ctfd.io/).**

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** Contact us via [the CTFd website](https://ctfd.io/contact/) for commercial support.

## Managed Hosting

Looking for managed CTFd deployments? Visit [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is tightly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on. Register your event with MajorLeagueCyber for features like automatic user login, score tracking, writeup submission, and event notifications.

To integrate, register an account, create an event, and add the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
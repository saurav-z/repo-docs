# CTFd: The Open-Source Capture The Flag Platform

**Create and run engaging cybersecurity Capture The Flag (CTF) competitions with CTFd, a user-friendly and highly customizable platform.**  This README provides an overview to get you started. For more in-depth information, visit the [official CTFd repository](https://github.com/CTFd/CTFd).

![CTFd Logo](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true)

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a comprehensive suite of features designed to facilitate exciting and effective CTF events:

*   **Challenge Creation & Management:**
    *   Intuitive Admin Interface for easy challenge creation, categorization, and management.
    *   Dynamic Scoring challenges to adjust difficulty and points.
    *   Unlockable challenge support for progressive challenges.
    *   Extensible architecture via challenge plugins to support custom challenge types.
    *   Supports static and regular expression (regex) based flags.
    *   Custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads with options for server storage or Amazon S3-compatible backends.
    *   Ability to limit challenge attempts and hide challenges.
    *   Automatic brute-force protection to prevent unauthorized access.
*   **Competition Modes:**
    *   Supports individual and team-based competitions.
    *   Allows players to compete individually or form teams.
*   **Scoring and Leaderboard:**
    *   Automatic tie resolution on the leaderboard.
    *   Option to hide scores from public view.
    *   Score freezing at a specific time for final results.
    *   Scoregraphs for the top 10 teams.
    *   Team progress graphs.
*   **Content and Communication:**
    *   Markdown-based content management system.
    *   SMTP and Mailgun email support.
    *   Email confirmation and password reset support.
*   **Event Management:**
    *   Automated competition start and end times.
    *   Team management features, including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable with plugin and theme interfaces.
    *   Importing and exporting CTF data for archiving and backups.
*   **And much more...**

## Installation

Getting started with CTFd is straightforward:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script (requires `apt`) to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to customize settings.
3.  **Run:**  Use `python serve.py` or `flask run` to start in debug mode.

**Docker:**

CTFd offers convenient Docker images for quick deployment:

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose from the source repository: `docker compose up`

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and connect with the CTFd community:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or custom projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify CTF management with managed CTFd deployments: [CTFd Website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on.  Register your CTF event with MLC to enable features like automated login, score tracking, writeup submissions, and event notifications.

To integrate with MajorLeagueCyber, create an account, register an event, and add the client ID and secret into `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
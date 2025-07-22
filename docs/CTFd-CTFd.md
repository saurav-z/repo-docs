# CTFd: The Open-Source Capture The Flag Platform

[![CTFd CI Status](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting Status](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a flexible and easy-to-use open-source platform designed to host and manage your own Capture The Flag (CTF) competitions.**  ([View the original repository](https://github.com/CTFd/CTFd))

![CTFd Screenshot](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)

## Key Features

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom challenge plugins.
    *   Offers static and regex-based flags with custom flag plugins.
    *   Includes unlockable hints and file uploads with S3 compatibility.
    *   Provides challenge attempt limits and challenge hiding.
    *   Admin interface to configure all aspects of CTF challenges.
*   **Competition Modes:**
    *   Supports individual and team-based competitions.
    *   Allows users to compete solo or collaborate in teams.
*   **Scoring and Leaderboards:**
    *   Automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing at a specific time.
    *   Scoregraphs comparing the top teams.
    *   Team progress graphs for visualizing team performance.
*   **Content Management & Communication:**
    *   Markdown content management system for flexible challenge descriptions.
    *   SMTP and Mailgun email support, including email confirmation and password reset features.
    *   Automated competition start and end times.
*   **Team and User Management:**
    *   Team management features, including hiding and banning capabilities.
    *   User account management options.
*   **Customization:**
    *   Extensive plugin and theme interfaces for complete customization.
    *   Import and export CTF data for archiving and backups.
*   **Other Features:**
    *   Bruteforce protection.
    *   And much more to enhance your CTF experience!

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to fit your needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker:** Use the pre-built Docker image:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or use Docker Compose:

```bash
docker compose up
```

Refer to the [CTFd documentation](https://docs.ctfd.io/) for advanced [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

For commercial support or specific project needs, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in a managed CTFd deployment? Check out [the CTFd website](https://ctfd.io/) for details.

## MajorLeagueCyber Integration

CTFd is designed to integrate seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker.  MLC provides event scheduling, team tracking, and single sign-on.

Integrating your CTF event with MajorLeagueCyber allows users to log in automatically, track scores, submit writeups, and receive event notifications. To integrate, register an account and create an event on MajorLeagueCyber, then add the client ID and client secret in `CTFd/config.py` or via the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
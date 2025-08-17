# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the go-to open-source platform for running and managing your own Capture The Flag (CTF) competitions, offering a customizable and user-friendly experience for both organizers and participants.** ([View the original repository](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Intuitive Admin Interface:** Easily create and manage challenges, categories, hints, and flags.
    *   Dynamic scoring challenges
    *   Unlockable challenge support
    *   Challenge plugin architecture
    *   Static & Regex-based flags
    *   Custom flag plugins
    *   Unlockable hints
    *   File uploads to server or S3
    *   Challenge attempt limits & hiding
    *   Automatic brute-force protection
*   **Flexible Competition Types:** Support for individual and team-based competitions.
    *   Users can play individually or in teams.
*   **Robust Scoreboard:** Real-time scoreboard with automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing functionality.
*   **Engaging Visuals:** Scoregraphs to compare teams.
*   **Content Management:** Markdown support.
*   **Communication Tools:** SMTP & Mailgun email support.
    *   Email confirmation and password recovery.
*   **Automation:** Automated competition start and end times.
*   **Team Management:** Features for team organization.
*   **Customization:** Extensive plugin and theme interfaces.
*   **Data Management:** Import and export CTF data.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

**Docker:**

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

**Docker Compose:**

```bash
docker compose up
```

For detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide, consult the CTFd documentation.

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support.

For commercial support, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in a managed CTFd deployment? Visit [the CTFd website](https://ctfd.io/) for details.

## MajorLeagueCyber Integration

CTFd is integrated with [MajorLeagueCyber](https://majorleaguecyber.org/) (MLC), which offers event scheduling, team tracking, and single sign-on.  MLC allows CTF events to automatically login, track scores, and submit writeups.

To integrate with MajorLeagueCyber, register an account, create an event, and enter the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
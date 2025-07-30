# CTFd: The Open-Source Capture The Flag Framework

<p align="center">
  <img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="200">
</p>

**CTFd is a powerful, open-source platform designed to easily host and manage your own Capture The Flag (CTF) competitions.**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a comprehensive set of features to create engaging and customizable CTF experiences:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags through the admin interface.
    *   Support for dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Plugin architecture for custom challenge creation.
    *   Support for static and regex-based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads to server or S3-compatible backend.
    *   Limit challenge attempts and hide challenges.
    *   Automatic brute-force protection.
*   **Competition Modes:**
    *   Individual and team-based competitions.
    *   Allow users to compete individually or form teams.
*   **Scoreboard & Reporting:**
    *   Automatic tie resolution.
    *   Option to hide scores from the public.
    *   Freeze scores at a specific time.
    *   Scoregraphs for the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support.
    *   Email confirmation and password reset support.
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management features, including hiding and banning.
*   **Customization:**
    *   Extensive customization via [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archiving.

## Installation

Follow these steps to get CTFd up and running:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal.

**Docker:**

*   **Run with Docker (Auto-Generated):**  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run with Docker Compose:** `docker compose up` (from the source repository)

For detailed deployment options, consult the [CTFd Documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:**  For commercial support or special projects, contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, check out the [CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.  Integrating with MLC allows users to automatically log in, track their scores, submit writeups, and receive notifications.

To integrate:
1. Register an account and create an event on MajorLeagueCyber.
2. Install the client ID and client secret in the `CTFd/config.py` file or admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)

---
**[Back to the CTFd Repository](https://github.com/CTFd/CTFd)**
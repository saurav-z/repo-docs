# CTFd: The Customizable Capture The Flag Framework

[![](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true)](https://github.com/CTFd/CTFd)

CTFd is a powerful, open-source platform designed to easily create and manage your own Capture The Flag (CTF) competitions.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring, unlockable challenges, and challenge plugins for custom challenge types.
    *   Offers static and regex-based flags, along with custom flag plugins.
    *   Includes unlockable hints.
    *   Allows file uploads to the server or Amazon S3-compatible backends.
    *   Provides options to limit challenge attempts and hide challenges.
    *   Includes automatic brute-force protection.

*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or form teams.

*   **Scoring and Leaderboard:**
    *   Features a scoreboard with automatic tie resolution.
    *   Offers options to hide scores and freeze scores at a specific time.
    *   Generates scoregraphs comparing the top 10 teams and team progress graphs.

*   **Content Management & Communication:**
    *   Includes a Markdown content management system.
    *   Offers SMTP + Mailgun email support, with email confirmation and password reset functionality.

*   **Competition Control:**
    *   Provides automatic competition start and end times.
    *   Offers team management features including hiding and banning teams.

*   **Customization:**
    *   Extensive customization options via plugin and theme interfaces.  See the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) documentation.
    *   Importing and exporting CTF data for backups and archival.

*   **And much more...**

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

### Docker

*   **Using Docker Images:**

    `docker run -p 8000:8000 -it ctfd/ctfd`

*   **Using Docker Compose:**

    `docker compose up` (from the source repository)

*   **For detailed deployment options and a Getting Started guide, please see the [CTFd docs](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.**

## Live Demo

*   [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   For general support, visit the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/):
    [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
*   For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

*   Looking for a hassle-free CTFd experience? Explore [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

By registering your CTF event with MajorLeagueCyber, users can automatically log in, track their scores, submit writeups, and receive event notifications.

To integrate:
1. Register an account and create an event on MajorLeagueCyber.
2. Install the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

**For more information, visit the original CTFd repository: [https://github.com/CTFd/CTFd](https://github.com/CTFd/CTFd)**
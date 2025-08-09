# CTFd: The Ultimate Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and highly customizable open-source platform designed to easily host and manage your own Capture The Flag competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd

*   **Challenge Management:**
    *   Create and customize challenges with ease from the admin interface.
    *   Supports dynamic scoring, unlockable challenges, and custom challenge plugins.
    *   Includes static and regex-based flags.
    *   Allows file uploads to the server or an Amazon S3-compatible backend.
    *   Configure challenge attempts limits and challenge visibility.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Teams can be created, managed, hidden, and banned.
*   **Scoreboard & Gamification:**
    *   Automated tie resolution on the scoreboard.
    *   Option to hide or freeze scores.
    *   Scoregraphs comparing top teams and tracking progress.
*   **Content & Communication:**
    *   Built-in Markdown content management system.
    *   SMTP and Mailgun email support for notifications and password resets.
    *   Automated competition start and end times.
*   **Customization:**
    *   Highly customizable via [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Import and export CTF data for archiving.
*   **Other Features:**
    *   Bruteforce protection.
    *   Hint system with unlockable hints.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script for system dependencies.
2.  **Configure:** Modify the `CTFd/config.ini` file to your requirements.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to run in debug mode.

**Docker:**

*   Use the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or, use Docker Compose from the source repository: `docker compose up`

**Resources:**

*   [CTFd Documentation](https://docs.ctfd.io/)
*   [Deployment Options](https://docs.ctfd.io/docs/deployment/installation)
*   [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   For commercial support or custom projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd experience?  Check out [the CTFd website](https://ctfd.io/) for managed deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.  Integrating your CTF with MajorLeagueCyber enables automated user login, score tracking, write-up submissions, and event notifications.

To integrate, register an account and create an event on MajorLeagueCyber and then configure the client ID and secret in `CTFd/config.py` or within the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
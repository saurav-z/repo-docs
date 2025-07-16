# CTFd: The Open-Source Capture The Flag Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, open-source platform designed to make running Capture The Flag (CTF) competitions simple and customizable.** [(Back to the project)](https://github.com/CTFd/CTFd)

## Key Features

CTFd provides everything you need to host engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring for challenges.
    *   Unlockable challenge support with customizable dependencies.
    *   Plugin architecture for custom challenge types.
    *   Static and regular expression (Regex) based flags.
    *   Custom flag plugins for extended flexibility.
    *   Unlockable hints to guide participants.
    *   File uploads with support for server or Amazon S3-compatible backends.
    *   Challenge attempt limits and challenge hiding.
    *   Bruteforce protection.
*   **Competition Modes:**
    *   Individual and team-based competitions.
    *   Team management features including hiding and banning.
*   **Scoring and Leaderboard:**
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores from the public.
    *   Ability to freeze scores at a specific time.
    *   Scoregraphs comparing the top teams and team progress graphs.
*   **Content and Communication:**
    *   Markdown-based content management system for rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support for notifications, confirmation, and password reset.
*   **Event Management:**
    *   Automated competition starting and ending times.
*   **Customization & Integration:**
    *   Extensive customization through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archiving.

## Installation

Get started with CTFd by following these steps:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script to install system dependencies.
2.  **Configure:** Modify the `CTFd/config.ini` file to match your desired settings.
3.  **Run:** Start the development server with `python serve.py` or `flask run` in a terminal.

You can also use the pre-built Docker images:

`docker run -p 8000:8000 -it ctfd/ctfd`

Or use Docker Compose:

`docker compose up`

For detailed deployment options and a getting started guide, refer to the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd experience? Check out [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker. Register your CTF event with MLC for:

*   Automated user login.
*   Individual and team score tracking.
*   Writeup submission.
*   Event notifications.

To integrate, create an account and event on MajorLeagueCyber and enter the client ID and secret in `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
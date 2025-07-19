# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly and customizable Capture The Flag (CTF) platform, perfect for hosting cybersecurity competitions.**

[Visit the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd is designed to be feature-rich and easy to use, providing everything you need to run engaging CTFs.

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges, and challenge plugins for custom challenge types.
    *   Includes static and regex-based flags, with support for custom flag plugins.
    *   Offers file uploads to the server or an Amazon S3-compatible backend.
    *   Allows limiting challenge attempts and hiding challenges.
    *   Challenge support for multiple levels
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoring & Ranking:**
    *   Provides a scoreboard with automatic tie resolution.
    *   Allows hiding scores from the public or freezing scores at a specific time.
    *   Generates scoregraphs comparing team progress and top 10 teams.
*   **Content & Communication:**
    *   Includes a Markdown-based content management system.
    *   Offers SMTP + Mailgun email support, with email confirmation and password reset features.
*   **Competition Control:**
    *   Automated competition starting and ending times.
    *   Team management features, including hiding and banning.
*   **Customization:**
    *   Highly customizable through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Importing and exporting of CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script for system dependency installation (using `apt`).
2.  **Configure:** Modify `CTFd/config.ini` to your desired settings.
3.  **Run:** Start the server using `python serve.py` or `flask run` in a terminal for debug mode.

**Docker:**

*   Use the auto-generated Docker images: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose from the source repository: `docker compose up`

For detailed deployment instructions, refer to the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: https://demo.ctfd.io/

## Support and Community

*   For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
*   For commercial support or special projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, visit the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.

To integrate with MajorLeagueCyber:

1.  Register an account and create an event on MLC.
2.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
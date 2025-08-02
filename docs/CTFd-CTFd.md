# CTFd: The Open-Source Capture The Flag Platform

**CTFd is a powerful and customizable open-source platform designed to easily host and manage Capture The Flag (CTF) competitions.** ([Original Repository](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags directly from the admin interface.
    *   Supports dynamic scoring.
    *   Offers unlockable challenge support.
    *   Provides a plugin architecture for custom challenge types.
    *   Supports static and regex-based flags, and custom flag plugins.
    *   Includes unlockable hints.
    *   Enables file uploads to the server or an Amazon S3-compatible backend.
    *   Allows limiting challenge attempts and hiding challenges.
    *   Offers automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams.
*   **Scoring & Leaderboard:**
    *   Features a scoreboard with automatic tie resolution.
    *   Allows hiding scores.
    *   Offers score freezing at specific times.
    *   Provides scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Includes a Markdown content management system.
    *   Offers SMTP and Mailgun email support.
    *   Supports email confirmation and password reset functionality.
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management features including hiding and banning.
*   **Customization:**
    *   Extensive customization options using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Handling:**
    *   Supports importing and exporting CTF data for archival purposes.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies via apt.
2.  **Configure:** Modify `CTFd/config.ini` to match your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

**Docker:**

*   Run the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose from the source repository: `docker compose up`

For detailed installation and deployment instructions, please refer to the [CTFd documentation](https://docs.ctfd.io/) including the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd in action: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get community support via the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special project inquiries, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, visit [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on.  Register your CTF event with MajorLeagueCyber to enable automatic logins, score tracking, writeup submissions, and event notifications.

To integrate, create an account, set up an event, and add the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
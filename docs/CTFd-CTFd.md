# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and flexible open-source platform designed to help you easily host and manage your own Capture The Flag (CTF) competitions.**  ([View on GitHub](https://github.com/CTFd/CTFd))

## Key Features

CTFd offers a comprehensive set of features to create engaging and customizable CTF experiences:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags directly from the admin interface.
    *   Support for dynamic scoring, unlockable challenges, and custom challenge plugins.
    *   Static and Regex-based flags, with custom flag plugin support.
    *   Unlockable hints and file uploads (to server or S3-compatible backend).
    *   Challenge attempt limits and challenge hiding options.
    *   Automatic brute-force protection.
*   **Competition Modes:**
    *   Support for both individual and team-based competitions.
*   **Scoreboard & Reporting:**
    *   Automatic tie resolution.
    *   Optional score hiding and score freezing.
    *   Scoregraphs for the top 10 teams and team progress graphs.
*   **Content Management:**
    *   Markdown-based content management system for easy content creation.
*   **Communication & Notifications:**
    *   SMTP and Mailgun email support.
    *   Email confirmation and password recovery features.
*   **Competition Control:**
    *   Automatic competition starting and ending.
    *   Team management (hiding and banning).
*   **Customization & Extensibility:**
    *   Highly customizable through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Management:**
    *   Importing and exporting of CTF data.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script (requires apt) to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to adjust settings.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker:**

*   Run a pre-built Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or, use Docker Compose (from the source repository): `docker compose up`

**Documentation:**
*   Refer to the [CTFd docs](https://docs.ctfd.io/) for [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: https://demo.ctfd.io/

## Support

*   Get basic support and connect with other users through the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   For commercial support or custom project needs, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in a managed CTFd deployment? Visit the [CTFd website](https://ctfd.io/) to explore managed hosting options.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.  Register your CTF on MLC to allow users to log in automatically, track scores, and more.

To integrate, create an event on MajorLeagueCyber and add the client ID and secret to your `CTFd/config.py`:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
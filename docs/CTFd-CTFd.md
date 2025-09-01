# CTFd: The Open-Source Capture The Flag Platform

CTFd is a powerful and user-friendly platform designed to host and manage Capture The Flag (CTF) competitions with ease.  **(Check out the original repo on Github: [https://github.com/CTFd/CTFd](https://github.com/CTFd/CTFd))**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through an intuitive Admin Interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and challenge plugin architecture for advanced customization.
    *   Offers both static and regex-based flag types, along with custom flag plugins.
    *   Includes unlockable hints, file uploads (server or S3-compatible), and options to limit attempts and hide challenges.
    *   Integrates automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Reporting:**
    *   Features a comprehensive scoreboard with automatic tie resolution.
    *   Provides options to hide scores and freeze scores at a specific time.
    *   Offers scoregraphs comparing the top teams and team progress graphs.
*   **Content Management & Communication:**
    *   Includes a Markdown-based content management system.
    *   Offers SMTP + Mailgun email support (with confirmation and password reset features).
*   **Competition Control & Customization:**
    *   Automated competition starting and ending times.
    *   Team management features (hiding, banning).
    *   Highly customizable through plugin and theme interfaces ([plugin docs](https://docs.ctfd.io/docs/plugins/overview), [theme docs](https://docs.ctfd.io/docs/themes/overview)).
    *   Allows importing and exporting of CTF data for archival.
*   **And much more**: Explore our many features, from our team management to plugins!

## Getting Started:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies (using apt).
2.  **Configure:** Modify `CTFd/config.ini` to match your preferences.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

**Docker Usage:**

*   Run the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose: `docker compose up` (from the source repository)

For detailed installation and deployment options, refer to the [CTFd documentation](https://docs.ctfd.io/).  Check out the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide for further assistance.

## Live Demo:

Explore the platform firsthand at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support:

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:**  Contact us via [CTFd website](https://ctfd.io/contact/) for commercial support or special projects.

## Managed Hosting:

Looking for managed CTFd deployments? Visit [the CTFd website](https://ctfd.io/) for more details.

## MajorLeagueCyber Integration:

CTFd is integrated with MajorLeagueCyber (MLC), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

*   Register your CTF event with MajorLeagueCyber to enable features like automated login, score tracking, writeup submissions, and event notifications.
*   To integrate, create an event and install the client ID and client secret in `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits:

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
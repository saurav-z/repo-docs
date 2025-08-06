# CTFd: The Open-Source Capture The Flag Framework

![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)
![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a versatile and user-friendly open-source platform designed to help you create and manage your own Capture The Flag (CTF) competitions with ease.**  ([Back to Original Repo](https://github.com/CTFd/CTFd))

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom challenge plugins.
    *   Offers static & regex-based flags with custom flag plugins.
    *   Allows for unlockable hints, file uploads to the server or S3-compatible backends, challenge attempt limits, and challenge hiding.
*   **Competition Modes & Features:**
    *   Supports individual and team-based competitions.
    *   Provides a scoreboard with automatic tie resolution, and the option to hide or freeze scores.
    *   Includes scoregraphs to compare team performance.
*   **Content & Communication:**
    *   Markdown content management system for creating engaging content.
    *   SMTP and Mailgun email support with email confirmation and password recovery features.
*   **Admin & Customization:**
    *   Automatic competition start and end times.
    *   Team management features including hiding and banning.
    *   Highly customizable via a plugin and theme interface, see the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) docs.
    *   Importing and exporting of CTF data for archival.
*   **Other Features:**
    *   Automatic brute-force protection.
    *   And much more!

## Installation

1.  **Install Dependencies:**  `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to match your requirements.
3.  **Run:** Use `python serve.py` or `flask run` to run in debug mode.

**Docker Options:**

*   **Docker Run:**  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:** `docker compose up`  (from the source repository)

**For detailed deployment and getting started guides, please consult the [CTFd documentation](https://docs.ctfd.io/).**

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the community at the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support, contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF experience with managed CTFd deployments available at [https://ctfd.io/](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is seamlessly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

Register your CTF event with MajorLeagueCyber for automated user logins, score tracking, writeup submissions, and event notifications.

To integrate, register an account, create an event, and add the client ID and client secret in `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
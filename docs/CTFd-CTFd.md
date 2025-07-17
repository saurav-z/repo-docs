# CTFd: The Open-Source Capture The Flag Framework

[<img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="150">](https://github.com/CTFd/CTFd)

CTFd is a powerful and customizable open-source platform that empowers you to host engaging and educational Capture The Flag (CTF) competitions.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

*   **Challenge Creation & Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring and unlockable challenges.
    *   Utilize challenge plugins for custom challenges.
    *   Offers static and regex-based flags.
    *   Allows custom flag plugins.
    *   Provides unlockable hints.
    *   Includes file uploads to the server or an Amazon S3-compatible backend.
    *   Enables challenge attempt limits and challenge hiding.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Offers automatic tie resolution on the scoreboard.
    *   Allows hiding scores from the public and freezing scores at a specific time.
*   **Scoring & User Experience:**
    *   Provides scoreboards with automatic tie resolution.
    *   Includes scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Offers a Markdown content management system.
    *   Provides SMTP and Mailgun email support with email confirmation and password reset functionality.
*   **Event Management:**
    *   Automates competition start and end times.
    *   Includes team management, hiding, and banning features.
*   **Customization:**
    *   Extensive customization options using [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Management:**
    *   Allows importing and exporting CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script for system dependency installation using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to enter debug mode.

### Docker

You can also run CTFd using Docker. Use the following command:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or, utilize Docker Compose:

```bash
docker compose up
```

Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/):
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, consider managed CTFd deployments on [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is closely integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.  Integrate your CTF with MLC to allow users to automatically log in, track scores, submit writeups, and receive event notifications.

To integrate with MajorLeagueCyber, register an account, create an event, and insert the client ID and client secret in the `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)

---

**[Visit the original CTFd repository on GitHub](https://github.com/CTFd/CTFd)**
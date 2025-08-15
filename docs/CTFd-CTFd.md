# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly, highly customizable, and open-source Capture The Flag (CTF) framework perfect for cybersecurity enthusiasts, educational institutions, and competitive gaming.**

![CTFd is a CTF in a can.](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags via an intuitive admin interface.
    *   Supports dynamic scoring challenges.
    *   Includes unlockable challenge support.
    *   Offers a plugin architecture for creating your own custom challenge types.
    *   Supports static and regex-based flags.
    *   Custom flag plugins are supported.
    *   Supports unlockable hints.
    *   Allows file uploads to the server or Amazon S3-compatible backends.
    *   Includes challenge attempt limits and the ability to hide challenges.
    *   Provides automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or collaborate in teams.
*   **Scoreboard & Analytics:**
    *   Features a scoreboard with automatic tie resolution.
    *   Provides options to hide scores.
    *   Supports freezing scores at a specific time.
    *   Generates scoregraphs for the top 10 teams and team progress graphs.
*   **Content Management & Communication:**
    *   Includes a Markdown-based content management system.
    *   Supports SMTP and Mailgun email integration.
    *   Offers email confirmation and password recovery features.
    *   Provides automatic competition start and end times.
*   **Team & User Management:**
    *   Enables team management, hiding, and banning.
*   **Customization & Extensibility:**
    *   Highly customizable using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Supports importing and exporting CTF data for archival.
*   **And much more...**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies (using apt).
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker:**
*   Use the auto-generated Docker images: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or, use Docker Compose: `docker compose up` (from the source repository)

Consult the [CTFd docs](https://docs.ctfd.io/) for comprehensive [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: https://demo.ctfd.io/

## Support and Community

For basic support and discussions, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special project needs, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in using CTFd without infrastructure management? Check out [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.

By registering your CTF event with MajorLeagueCyber, users can enjoy automatic login, track scores, submit writeups, and receive event notifications.

To integrate with MajorLeagueCyber, create an account, set up an event, and enter the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

[Visit the original repository on GitHub](https://github.com/CTFd/CTFd)
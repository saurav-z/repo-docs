# CTFd: The Premier Capture The Flag (CTF) Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

CTFd is an open-source, highly customizable Capture The Flag (CTF) platform designed to make running and participating in CTFs easier and more engaging. **Looking to host your own CTF? CTFd provides everything you need with a focus on ease of use and flexibility.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges, and custom challenge types via a plugin architecture.
    *   Includes static and regex-based flags, along with custom flag plugins.
    *   Allows for file uploads, challenge attempt limits, and challenge hiding.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Ranking:**
    *   Automatic tie resolution.
    *   Ability to hide scores and freeze them at specific times.
    *   Scoregraphs to compare team progress.
*   **Content Management:**
    *   Built-in Markdown support for rich content creation.
*   **Communication & Notifications:**
    *   SMTP and Mailgun email support (including confirmation and password reset).
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management features including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable via [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Data importing and exporting for archiving.
*   **And Much More...**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Consider using the `prepare.sh` script to install system dependencies using apt.
2.  **Configure CTFd:** Modify `CTFd/config.ini` to match your needs.
3.  **Run the Application:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker:**

*   Run the auto-generated Docker image:  `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose: `docker compose up` (from the source repository).

**Resources:**

*   [CTFd Documentation](https://docs.ctfd.io/) for detailed [deployment instructions](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Try out CTFd yourself with this live demo:  https://demo.ctfd.io/

## Support & Community

Get support and connect with other CTFd users:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or special project inquiries, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For those who prefer managed CTFd deployments, visit [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is closely integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker. Register your CTF event with MajorLeagueCyber to leverage features such as:

*   Automated user login.
*   Individual and team score tracking.
*   Writeup submissions.
*   Event notifications.

To integrate, register an account, create an event, and add your client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo: [Laura Barbera](http://www.laurabb.com/)
*   Theme: [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound: [Terrence Martin](https://soundcloud.com/tj-martin-composer)
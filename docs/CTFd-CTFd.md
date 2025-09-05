# CTFd: The Ultimate Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and flexible Capture The Flag (CTF) platform that makes running and managing CTF events a breeze.**  Check out the [original repository](https://github.com/CTFd/CTFd).

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags via an intuitive Admin Interface.
    *   Support for dynamic scoring challenges to keep things interesting.
    *   Unlockable challenge support for progressive learning.
    *   Extensible challenge plugin architecture for customized challenges.
    *   Supports static & regex-based flags with custom flag plugins.
    *   Offer unlockable hints to guide players.
    *   File uploads to the server or Amazon S3-compatible backend.
    *   Limit challenge attempts & hide challenges for strategic control.
*   **Competition & Team Features:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or collaborate in teams.
    *   Provides a scoreboard with automatic tie resolution.
    *   Option to hide scores from the public or freeze them at a specific time.
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system for rich challenge descriptions and announcements.
    *   SMTP & Mailgun email support for notifications and password recovery.
    *   Email confirmation for secure account management.
    *   "Forgot password" functionality for user convenience.
*   **Event & Team Administration:**
    *   Automated competition start and end times.
    *   Team management features, including hiding and banning.
*   **Customization & Integration:**
    *   Fully customizable via [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archiving and backups.
*   **And much more:** Explore the full feature set to unlock your CTF potential!

## Getting Started with CTFd

### Installation

1.  **Install Dependencies:**  `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script for system dependencies via `apt`.
2.  **Configure:** Modify the `CTFd/config.ini` file to customize your CTF.
3.  **Run the Server:** Use `python serve.py` or `flask run` to launch in debug mode.

### Docker Deployment

Use pre-built Docker images for quick setup:

`docker run -p 8000:8000 -it ctfd/ctfd`

Alternatively, deploy with Docker Compose:

`docker compose up`  (from the source repository directory)

### Further Information

*   Explore detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide in the CTFd documentation.
*   [Live Demo](https://demo.ctfd.io/)

## Support & Community

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support and discussions.
*   For commercial support or special project assistance, please [contact us](https://ctfd.io/contact/).
*   **Managed Hosting:** Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/) for ease of use.

## Integration with MajorLeagueCyber (MLC)

CTFd seamlessly integrates with MajorLeagueCyber (MLC), a CTF stats tracker. MLC provides features like:

*   Event scheduling.
*   Team tracking.
*   Single sign-on for CTF events.

To integrate, create an MLC account, create an event, and add the client ID and secret to `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
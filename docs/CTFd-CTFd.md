# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a flexible and user-friendly Capture The Flag framework designed to help you create and manage engaging cybersecurity competitions.**

[Visit the original repository](https://github.com/CTFd/CTFd)

## Key Features of CTFd

CTFd offers a wide array of features designed to make CTF creation and management a breeze:

*   **Intuitive Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges.
    *   Unlockable challenge functionality.
    *   Built-in support for custom challenge plugins.
    *   Supports static & regex-based flags.
    *   Custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads to your server or an Amazon S3-compatible backend.
    *   Challenge attempt limits and challenge hiding options.
    *   Automatic brute-force protection.
*   **Competition Management:**
    *   Supports individual and team-based competitions.
    *   Automatic tie resolution in the scoreboard.
    *   Option to hide scores from the public and freeze scores at a specific time.
*   **User Experience:**
    *   Scoreboards with clear leaderboards and team progress graphs.
    *   Markdown content management system for creating engaging content.
    *   SMTP and Mailgun email integration, including email confirmation and password recovery.
    *   Automatic competition start and end times.
    *   Team management tools (hiding and banning).
*   **Customization:**
    *   Extensive plugin and theme support for complete customization.
    *   Importing and exporting CTF data for backups and archives.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to customize settings.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker Deployment:**

*   Use the auto-generated Docker images: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose: `docker compose up` (from the source repository)

For detailed deployment options, refer to the [CTFd docs](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:** Contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/) for custom projects or premium support.

## Managed Hosting

Simplify your CTF experience with managed CTFd deployments: [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
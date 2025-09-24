# CTFd: The Premier Capture The Flag (CTF) Framework

**CTFd is a powerful and customizable open-source platform that makes it easy to create and run your own Capture The Flag (CTF) competitions.** Explore the original repository on [GitHub](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Easy Challenge Creation:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Support for dynamic scoring challenges, and unlockable challenges.
    *   Plugin architecture for custom challenge types.
    *   Static & Regex-based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads (server or S3).
    *   Challenge attempt limits and challenge hiding.
    *   Automatic brute-force protection.
*   **Team and Individual Competitions:**
    *   Support for both individual and team-based competitions.
*   **Robust Scoring & Scoreboard:**
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing capabilities.
    *   Score graphs comparing top teams.
    *   Team progress graphs.
*   **Content Management & Communication:**
    *   Markdown-based content management.
    *   SMTP and Mailgun email support (confirmation, password reset).
*   **Competition Management:**
    *   Automated start and end times.
    *   Team management (hiding, banning).
*   **Extensible & Customizable:**
    *   Customize everything using plugins and themes.
    *   Import and export CTF data.
*   **And much more!**

## Getting Started with CTFd

Follow these steps to get CTFd up and running:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` to start the server.

**Docker:** Use pre-built Docker images:

    `docker run -p 8000:8000 -it ctfd/ctfd`

**Docker Compose:**

    `docker compose up` (from source repository)

**Documentation:** Comprehensive documentation can be found on the [CTFd Docs](https://docs.ctfd.io/) including [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand at: https://demo.ctfd.io/

## Support & Community

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support.
*   For commercial support, contact us via the [CTFd website](https://ctfd.io/contact/).

## Managed Hosting

Looking for a managed CTFd solution? Visit the [CTFd website](https://ctfd.io/) for managed deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

To integrate:

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
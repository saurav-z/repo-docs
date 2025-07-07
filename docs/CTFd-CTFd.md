# CTFd: The Ultimate Capture The Flag (CTF) Framework

[![CTFd CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

**CTFd is a user-friendly and highly customizable Capture The Flag framework designed to help you host engaging and educational cybersecurity competitions.** ([See the original repo](https://github.com/CTFd/CTFd) for more details.)

## Key Features

CTFd offers a comprehensive suite of features to create, manage, and run your CTF events:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags directly from the admin interface.
    *   Dynamic scoring challenges and unlockable challenge support.
    *   Plugin architecture for custom challenges.
    *   Supports static and regex-based flags, plus custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads to server or S3-compatible backends.
    *   Challenge attempt limiting and hiding capabilities.
*   **Competition Structure:**
    *   Supports both individual and team-based competitions.
*   **Scoring & Ranking:**
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores from the public.
    *   Score freezing at specific times.
    *   Scoregraphs for the top 10 teams and individual team progress.
*   **Content & Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support (including email confirmation and password reset).
*   **Event Control:**
    *   Automatic competition starting and ending.
    *   Team management: hiding, banning, and more.
*   **Customization & Integration:**
    *   Extensive customization using plugins and themes.
    *   Importing and exporting of CTF data for archiving.
*   **And More!**
## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can use the `prepare.sh` script for system dependencies (apt).
2.  **Configure:** Modify `CTFd/config.ini` to match your needs.
3.  **Run:**  Use `python serve.py` or `flask run` for debug mode.

**Docker:**

*   Run pre-built image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose: `docker compose up` (from the source repository)

See the [CTFd documentation](https://docs.ctfd.io/) for deployment and [getting started](https://docs.ctfd.io/tutorials/getting-started/) guides.

## Live Demo

Try out a live demo at [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us via [CTFd website](https://ctfd.io/contact/) for commercial support or special projects.

## Managed Hosting

For a hassle-free CTFd experience, explore managed deployments at [CTFd website](https://ctfd.io/).

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

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
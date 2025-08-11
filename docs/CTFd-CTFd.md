# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the leading open-source platform for creating and running your own Capture The Flag (CTF) competitions, offering unparalleled ease of use and customization.**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)
[![](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true)](https://github.com/CTFd/CTFd)

CTFd provides a robust and flexible framework for hosting CTFs, enabling you to engage and challenge cybersecurity enthusiasts of all skill levels.

## Key Features:

*   **Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Dynamic scoring challenges and unlockable challenge support.
    *   Plugin architecture for creating custom challenges.
    *   Static and Regex-based flag options, with custom flag plugins.
    *   Unlockable hints to guide participants.
    *   File uploads to the server or an Amazon S3-compatible backend.
    *   Challenge attempt limits and the ability to hide challenges.
*   **Competition Modes:**
    *   Individual and team-based competition options.
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores.
    *   Freeze scores at a specific time.
*   **Scoreboard & Reporting:**
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management:**
    *   Markdown content management system for rich content.
*   **Communication:**
    *   SMTP and Mailgun email support.
    *   Email confirmation and password recovery features.
*   **Automation & Management:**
    *   Automatic competition start and end times.
    *   Team management including hiding and banning.
*   **Customization:**
    *   Extensive customization via the plugin and theme interfaces (see [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) documentation).
*   **Data Management:**
    *   Import and export CTF data for archival.
*   **And much more...**

## Installation

**Prerequisites:** Python and pip installed.

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script for system dependency installation via apt.
2.  **Configure:** Modify `CTFd/config.ini` to your specific requirements.
3.  **Run:** Use `python serve.py` or `flask run` to start the server in debug mode.

**Docker:**

*   Use pre-built Docker images with: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Alternatively, use Docker Compose: `docker compose up` (from the source repository)

**Documentation:** For detailed deployment options and a getting started guide, refer to the official [CTFd docs](https://docs.ctfd.io/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us via [ctfd.io/contact/](https://ctfd.io/contact/) for commercial support or custom projects.
*   **Managed Hosting:** Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on. Integrate with MajorLeagueCyber for features such as automatic user login, team scores tracking, writeup submissions, and event notifications.

*   Register an account and create an event on MajorLeagueCyber.
*   Insert your client ID and client secret into `CTFd/config.py` or the admin panel:
    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

**[View the CTFd Repository on GitHub](https://github.com/CTFd/CTFd)**
# CTFd: The Open-Source Capture The Flag Framework

**CTFd is a powerful and flexible open-source platform designed to host engaging and customizable Capture The Flag (CTF) competitions.**  [Explore the original repository on GitHub](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd empowers you to create dynamic and engaging CTF experiences with a wealth of features:

*   **Challenge Creation & Management:**
    *   Intuitive Admin Interface to create challenges, categories, hints, and flags.
    *   Dynamic Scoring Challenges to keep the competition exciting.
    *   Support for unlockable challenges.
    *   Challenge plugin architecture for custom challenge types.
    *   Support for Static & Regex based flags.
    *   Custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads to the server or Amazon S3-compatible backends.
    *   Challenge attempt limiting & challenge hiding.
*   **Competition Modes & Team Management:**
    *   Supports both individual and team-based competitions.
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores from the public.
    *   Score freezing at a specific time.
    *   Scoregraphs comparing top teams and team progress.
    *   Team management features, including hiding and banning.
*   **Content & Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support for notifications and password resets.
    *   Automatic competition starting and ending times.
*   **Customization & Integration:**
    *   Highly customizable through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Importing and exporting of CTF data.
*   **Additional Features:**
    *   Bruteforce protection
    *   And much more!

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to drop into debug mode.

**Docker:**

*   Use auto-generated Docker images:  `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose:  `docker compose up` (from the source repository)

Refer to the [CTFd docs](https://docs.ctfd.io/) for [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and connect with the CTFd community:

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** For commercial support or custom projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF deployment with managed CTFd hosting: [CTFd website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker, providing features like event scheduling, team tracking, and single sign-on. Register your CTF event to enable automatic user logins, score tracking, writeup submissions, and event notifications.

To integrate with MajorLeagueCyber:

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py` or the admin panel.

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
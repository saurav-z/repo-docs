# CTFd: The Open-Source Capture The Flag Platform

CTFd is a powerful and easy-to-use platform designed for creating and running Capture The Flag (CTF) competitions. ([View on GitHub](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

CTFd is designed to be user-friendly and highly customizable, empowering you to create engaging and challenging CTF experiences. Here are some of its key features:

*   **Challenge Creation and Management:**
    *   Create challenges, categories, hints, and flags directly through the admin interface.
    *   Supports dynamic scoring to adjust challenge difficulty.
    *   Offers unlockable challenge support to guide players.
    *   Provides a robust plugin architecture for custom challenges.
    *   Supports both static and regular expression (regex)-based flags.
    *   Enables custom flag plugins for advanced flag types.
    *   Includes unlockable hints to assist participants.
    *   Allows file uploads to the server or integration with Amazon S3-compatible backends.
    *   Allows you to limit challenge attempts.
    *   Ability to hide challenges.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete individually or form teams.
*   **Scoreboard and Reporting:**
    *   Features an automatic tie resolution scoreboard.
    *   Offers the option to hide scores from the public.
    *   Allows score freezing at specific times to control the competition's pace.
    *   Provides scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content and Communication:**
    *   Includes a Markdown content management system for rich content.
    *   Offers SMTP and Mailgun email support.
    *   Supports email confirmation and password reset functionality.
*   **Competition Control:**
    *   Provides automated competition starting and ending times.
    *   Allows for team management, hiding, and banning.
*   **Customization and Extensibility:**
    *   Highly customizable via [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Allows importing and exporting CTF data for archiving and backups.
*   **Additional Features:**
    *   Automatic bruteforce protection.
    *   And more...

## Installation

1.  **Install Dependencies:** Run `pip install -r requirements.txt`.  You can also use the `prepare.sh` script (if you have `apt`) to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to adjust settings as needed.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

You can also use Docker images:

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or with Docker Compose: `docker compose up` (from the source repository)

Refer to the [CTFd docs](https://docs.ctfd.io/) for deployment options and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd with a live demo: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support.
*   Contact us for commercial support or specialized projects: [https://ctfd.io/contact/](https://ctfd.io/contact/)

## Managed Hosting

For managed CTFd deployments, visit: [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.

Registering your CTF event allows users to log in automatically, track scores, submit writeups, and receive notifications.

To integrate, register an account, create an event, and install the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
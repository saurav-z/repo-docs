# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and customizable open-source Capture The Flag (CTF) framework designed to make hosting your own cybersecurity competitions easy and engaging.**  [View the source on GitHub](https://github.com/CTFd/CTFd).

## Key Features of CTFd

*   **Easy Challenge Creation and Management:**
    *   Create challenges, categories, hints, and flags directly from the admin interface.
    *   Supports dynamic scoring challenges.
    *   Offers unlockable challenge support.
    *   Utilizes a plugin architecture for custom challenge types.
    *   Supports static and regex-based flags, plus custom flag plugins.
    *   Includes unlockable hints.
    *   Allows file uploads to the server or S3-compatible backends.
    *   Offers options to limit challenge attempts and hide challenges.
    *   Provides automatic brute-force protection.

*   **Team and Individual Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or collaborate in teams.

*   **Interactive Scoreboard and Progress Tracking:**
    *   Features a scoreboard with automatic tie resolution.
    *   Offers the option to hide scores from the public or freeze them at a specific time.
    *   Provides score graphs to compare the top 10 teams and individual team progress.

*   **Content Management and Communication Tools:**
    *   Includes a Markdown content management system.
    *   Offers SMTP and Mailgun email support, with email confirmation and password reset features.
    *   Provides automatic competition start and end times.

*   **Advanced Features:**
    *   Team management, hiding, and banning capabilities.
    *   Extensive customization via plugin and theme interfaces ([plugins](https://docs.ctfd.io/docs/plugins/overview), [themes](https://docs.ctfd.io/docs/themes/overview)).
    *   Importing and exporting of CTF data for archiving.
    *   And much more!

## Installation

1.  **Install Dependencies:**  `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies with apt.
2.  **Configure:** Modify `CTFd/config.ini` to your specific needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker Options:**

*   **Run a pre-built image:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Use Docker Compose:**  `docker compose up` (from the source repository).

For detailed installation and deployment instructions, see the [CTFd documentation](https://docs.ctfd.io/) including the [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Check out a live demo at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/) for commercial support or custom projects.
*   **Managed Hosting:**  Consider [the CTFd website](https://ctfd.io/) for managed CTFd deployments if you prefer not to manage the infrastructure yourself.

## MajorLeagueCyber Integration

CTFd is tightly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/).  MLC provides event scheduling, team tracking, and single sign-on for CTF events.

To integrate with MajorLeagueCyber:

1.  Register an account on MLC.
2.  Create an event.
3.  Install the client ID and secret in `CTFd/config.py` (or the admin panel):

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
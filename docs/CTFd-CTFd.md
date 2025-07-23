# CTFd: The Premier Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is an open-source, easy-to-use, and highly customizable Capture The Flag (CTF) framework that empowers you to create engaging cybersecurity competitions.**

[Visit the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd provides a comprehensive platform for running CTFs, packed with features to make your event a success:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags via the admin interface.
    *   Support for dynamic scoring, unlockable challenges, and custom challenge types.
    *   Utilize static and regex-based flags, along with custom flag plugins.
    *   Offer unlockable hints to guide participants.
    *   Enable file uploads to the server or Amazon S3-compatible backends.
    *   Control challenge attempts and hide challenges as needed.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allow users to compete solo or form teams.
*   **Scoring and Leaderboards:**
    *   Real-time scoreboard with automatic tie resolution.
    *   Option to hide scores from the public or freeze them at a specific time.
    *   Scoregraphs comparing top teams and team progress graphs.
*   **Content Management & Communication:**
    *   Markdown content management system for rich content.
    *   SMTP and Mailgun email support for notifications and password recovery.
    *   Automatic competition start and end times.
*   **User and Team Management:**
    *   Team management tools, including hiding and banning capabilities.
*   **Customization & Extensibility:**
    *   Highly customizable through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archival and backup.
*   **And much more!**

## Installation

To get started with CTFd:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to your desired settings.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

**Docker:**

*   Run a Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose:  `docker compose up` (from the source repository)

For detailed deployment instructions, consult the [CTFd documentation](https://docs.ctfd.io/docs/deployment/installation) and [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Explore a live demo of CTFd:  [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and support from the community:

*   **MajorLeagueCyber Community:**  [https://community.majorleaguecyber.org/](https://community.majorleaguecyber.org/)
*   **Commercial Support:** Contact the team through the [CTFd website](https://ctfd.io/contact/) for special projects or commercial support.

## Managed Hosting

For a hassle-free experience, consider managed CTFd deployments through the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker that provides event scheduling, team tracking, and single sign-on.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Insert the client ID and client secret into `CTFd/config.py` or the admin panel:
    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
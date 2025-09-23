# CTFd: The Open-Source Capture The Flag Framework

Easily create and manage your own Capture The Flag (CTF) competitions with CTFd, a versatile and customizable framework.  Learn more and contribute at the [original repository](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd empowers you to build engaging CTF experiences with these key features:

*   **Challenge Management:**
    *   Create and customize challenges, categories, hints, and flags from the admin interface.
    *   Support for dynamic scoring, unlockable challenges, and custom flag plugins.
    *   Configure file uploads (server or S3-compatible backend).
    *   Set challenge attempt limits and hide challenges.
*   **Competition Structure:**
    *   Supports both individual and team-based competitions.
    *   Automatic tie resolution on the scoreboard.
    *   Option to hide scores or freeze them at a specific time.
    *   Scoregraphs for team comparisons and progress tracking.
*   **Content & Communication:**
    *   Markdown content management system for creating rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support, including confirmation and password reset features.
    *   Automated competition start and end times.
*   **Admin & User Management:**
    *   Team management, hiding, and banning capabilities.
*   **Customization & Integration:**
    *   Extend functionality using plugins and themes (plugin and theme interfaces).
    *   Import and export CTF data for archiving and backups.
*   **Bruteforce protection.**
*   **Much more...**

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to your specific needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

### Deployment

*   **Docker:**
    *   Use the pre-built Docker images: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Alternatively, use Docker Compose: `docker compose up`

Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand at the [live demo](https://demo.ctfd.io/).

## Support

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   For commercial support or specialized projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in a managed CTFd deployment? Visit the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is closely integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF statistics and event management platform.  Integrate your CTF with MLC to enable features like single sign-on, team tracking, and writeup submissions.

To integrate:

1.  Register an account with MajorLeagueCyber.
2.  Create an event.
3.  Add the client ID and client secret to your `CTFd/config.py` or via the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
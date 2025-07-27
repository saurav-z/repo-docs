# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/mysql.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, user-friendly, and highly customizable open-source framework designed to host and manage your own Capture The Flag (CTF) competitions.**

[<img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="150"/>](https://github.com/CTFd/CTFd)

## Key Features

CTFd offers a comprehensive suite of features to create and run engaging CTF events:

*   **Challenge Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges, and custom challenge types via a plugin architecture.
    *   Includes static and regex-based flags, along with custom flag plugins.
    *   Offers unlockable hints, file uploads (server or S3-compatible), and challenge attempt limits with hiding options.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Team management capabilities (hiding and banning).
*   **Scoring & Leaderboards:**
    *   Automatic tie resolution.
    *   Ability to hide scores and freeze them at a specific time.
    *   Scoregraphs and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system for creating rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support with email confirmation and password reset functionality.
*   **Competition Control:**
    *   Automatic competition start and end times.
*   **Customization:**
    *   Highly customizable using the plugin and theme interfaces.
*   **Data Management:**
    *   Import and export CTF data for archiving and reuse.
*   **And Much More:** Explore the comprehensive feature set to create compelling CTF events.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF.
3.  **Run:** Start the server using `python serve.py` or `flask run` (debug mode).

**Docker:**

*   **Run with Docker (Quick Start):** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run with Docker Compose:** `docker compose up` (from the source repository)

For detailed deployment options and a getting started guide, consult the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us for commercial support and custom project inquiries via [CTFd website](https://ctfd.io/contact/).
*   **Managed Hosting:** Discover managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## Integration with MajorLeagueCyber

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/) for enhanced event management. MLC provides:

*   Event Scheduling
*   Team Tracking
*   Single Sign-On (SSO)

**Integration Steps:**

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and secret in `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

**[Visit the CTFd GitHub Repository](https://github.com/CTFd/CTFd) to get started today!**
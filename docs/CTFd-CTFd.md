# CTFd: The Customizable Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is an open-source platform designed to make it easy to create and manage your own Capture The Flag (CTF) competitions.** ([View the original repository](https://github.com/CTFd/CTFd))

## Key Features of CTFd:

*   **Challenge Creation and Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges and hints.
    *   Utilizes a plugin architecture for custom challenge types and flag plugins.
    *   Offers static and regex-based flag validation.
    *   Allows file uploads to the server or an S3-compatible backend.
    *   Provides options to limit challenge attempts and hide challenges.
    *   Includes automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or form teams.
*   **Scoreboard and Rankings:**
    *   Provides a comprehensive scoreboard with automatic tie resolution.
    *   Offers options to hide scores or freeze them at a specific time.
    *   Includes scoregraphs and team progress graphs for analysis.
*   **Content Management and Communication:**
    *   Features a Markdown content management system.
    *   Offers SMTP and Mailgun email support (with confirmation and password reset).
    *   Allows automatic competition starting and ending.
*   **Team and User Management:**
    *   Includes team management features (hiding and banning).
*   **Customization:**
    *   Highly customizable using plugin and theme interfaces.
*   **Data Import/Export:**
    *   Supports importing and exporting CTF data for backups and archives.
*   **And much more!**

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure CTFd:** Modify `CTFd/config.ini` to match your desired settings.
3.  **Run the application:** Use `python serve.py` or `flask run` in a terminal to run in debug mode.

**Docker:**

*   **Run with Docker:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Use Docker Compose:** `docker compose up` (from the source repository).

For detailed deployment instructions and configuration options, please refer to the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support and special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd experience? Explore [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

To integrate, register your CTF event with MajorLeagueCyber and install the client ID and secret in the `CTFd/config.py` or admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
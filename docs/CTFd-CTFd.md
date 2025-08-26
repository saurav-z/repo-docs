# CTFd: The Open-Source Capture The Flag Framework

**CTFd is a powerful and user-friendly framework designed to create and manage Capture The Flag (CTF) competitions with ease.** Learn more and contribute to CTFd on [GitHub](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd offers a comprehensive set of features to create engaging and customizable CTF experiences:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges and unlockable challenge support.
    *   Offers a plugin architecture for custom challenge development.
    *   Allows for static & regex-based flags and custom flag plugins.
    *   Includes unlockable hints and file uploads to the server or an S3-compatible backend.
    *   Provides options to limit challenge attempts and hide challenges.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Enables users to play solo or form teams.
*   **Scoreboard & Progress Tracking:**
    *   Features a scoreboard with automatic tie resolution.
    *   Allows hiding scores from the public and freezing scores at a specific time.
    *   Generates scoregraphs comparing top teams and team progress graphs.
*   **Content Management & Communication:**
    *   Includes a Markdown content management system.
    *   Provides SMTP + Mailgun email support (including email confirmation and password reset).
    *   Supports automatic competition starting and ending.
*   **User & Team Management:**
    *   Offers team management features (hiding and banning).
*   **Customization & Integration:**
    *   Customize with plugins and themes using an intuitive interface.
    *   Supports importing and exporting CTF data for archival.
*   **Additional Features:**
    *   Automatic brute-force protection.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your specific needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker:**

*   **Run the auto-generated Docker image:**
    `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Use Docker Compose:**
    `docker compose up` (from the source repository)

**Important Resources:**

*   [CTFd Documentation](https://docs.ctfd.io/) for in-depth information.
*   [Deployment Options](https://docs.ctfd.io/docs/deployment/installation)
*   [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support through:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   For commercial support or special projects, contact us: [https://ctfd.io/contact/](https://ctfd.io/contact/)

## Managed Hosting

For managed CTFd deployments, visit: [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker. Integrate your CTF with MLC to enable automatic login, team tracking, writeup submission, and event notifications.

To integrate, register for an account, create an event, and add the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
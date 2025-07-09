# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and flexible open-source framework designed to help you create and manage your own Capture The Flag (CTF) competitions.** ([Original Repository](https://github.com/CTFd/CTFd))

## Key Features

CTFd offers a comprehensive set of features to create engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags directly through the admin interface.
    *   Supports dynamic scoring challenges and unlockable challenge support.
    *   Plugin architecture allows for custom challenge creation.
    *   Supports static and regex-based flags, and custom flag plugins.
    *   Offers unlockable hints and file uploads.
    *   Allows limiting challenge attempts and hiding challenges.
    *   Includes automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Provides options for users to compete individually or form teams.
*   **Scoreboard & Reporting:**
    *   Includes a scoreboard with automatic tie resolution.
    *   Offers options to hide scores or freeze scores at a specific time.
    *   Features scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Integrates a Markdown content management system.
    *   Offers SMTP and Mailgun email support, including email confirmation and password recovery.
*   **Event Management:**
    *   Automated competition starting and ending.
    *   Team management features, including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable via plugin and theme interfaces.  Check out the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) documentation.
*   **Data Management:**
    *   Supports importing and exporting CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script (requires apt) to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to fit your needs.
3.  **Run:** Use `python serve.py` or `flask run` to run the server in debug mode.

**Docker:**

*   Run with a pre-built image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose: `docker compose up` (from the source repository directory).

Refer to the [CTFd documentation](https://docs.ctfd.io/) for deployment options and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd's capabilities at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for general support and discussions.
*   **Commercial Support:** For commercial support or specific project needs, contact the CTFd team via [contact form](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments without the infrastructure management, visit the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker. MLC provides event scheduling, team tracking, and single sign-on capabilities. Register your CTF event with MLC to enable automatic logins, score tracking, writeup submission, and event notifications.

To integrate, register an account and create an event on MajorLeagueCyber. Then, enter the client ID and secret into `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
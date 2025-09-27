# CTFd: The Open-Source Capture The Flag Platform

**CTFd is the premier, open-source platform designed to simplify the creation and management of Capture The Flag (CTF) competitions, empowering both beginners and seasoned cybersecurity professionals.** ([View the original repo](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a comprehensive suite of features to facilitate engaging and educational CTF events:

*   **Challenge Management:**
    *   Create custom challenges with dynamic scoring.
    *   Implement unlockable challenge support for a structured experience.
    *   Leverage a plugin architecture to build your own custom challenge types.
    *   Supports static and regex-based flags, with custom flag plugins.
    *   Implement unlockable hints to guide players.
    *   Enable file uploads, with options for server storage or Amazon S3-compatible backends.
    *   Limit challenge attempts and hide challenges for a more controlled environment.
    *   Automatic brute-force protection
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allow users to compete solo or collaborate in teams.
*   **Scoring and Ranking:**
    *   Provides a dynamic scoreboard with automatic tie resolution.
    *   Option to hide scores from the public for added suspense.
    *   Ability to freeze scores at a specific time.
    *   Generates scoregraphs comparing top teams and individual team progress graphs.
*   **Content and Communication:**
    *   Utilizes a Markdown content management system for flexible content creation.
    *   Offers SMTP and Mailgun email support, including email confirmation and password reset functionality.
*   **Administration & Customization:**
    *   Control competition start and end times automatically.
    *   Manage teams, including hiding and banning options.
    *   Extensive customization through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archival purposes.
*   **And much more:** Explore features such as team management and banning.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF instance.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker:**

You can quickly get started with Docker:

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose: `docker compose up` (from the source repository)

Comprehensive [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide are available in the CTFd documentation.

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For community support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or specialized project assistance, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For those seeking a hassle-free CTF experience, explore [CTFd's managed hosting](https://ctfd.io/) options.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on.

To integrate with MajorLeagueCyber, register an account, create an event, and insert the client ID and client secret in the appropriate section of `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
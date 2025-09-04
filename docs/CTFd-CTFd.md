# CTFd: The Ultimate Capture The Flag (CTF) Framework 🏆

CTFd is a powerful and customizable open-source framework designed to help you easily host and manage your own Capture The Flag (CTF) competitions. ([Check out the original repository](https://github.com/CTFd/CTFd))

![CTFd is a CTF in a can.](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features: Run Your Own CTF With Ease

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges.
    *   Includes unlockable challenge support.
    *   Leverage a challenge plugin architecture to create custom challenge types.
    *   Supports static and regex-based flags.
    *   Custom flag plugins available.
    *   Provides unlockable hints.
    *   Allows file uploads to server or S3-compatible backend.
    *   Includes options to limit challenge attempts and hide challenges.
    *   Automatic brute-force protection.
*   **Competition Modes:**
    *   Supports individual and team-based competitions.
    *   Allows users to play solo or in teams.
*   **Scoring & Leaderboards:**
    *   Features a scoreboard with automatic tie resolution.
    *   Offers options to hide scores from the public.
    *   Allows freezing scores at a specific time.
    *   Generates scoregraphs comparing the top 10 teams and individual team progress graphs.
*   **Content & Communication:**
    *   Includes a Markdown content management system.
    *   Supports SMTP + Mailgun email integration.
    *   Provides email confirmation and password recovery features.
*   **Event Management:**
    *   Automated competition starting and ending times.
    *   Team management tools (hide/ban).
*   **Customization & Extensibility:**
    *   Extensive plugin ([plugins documentation](https://docs.ctfd.io/docs/plugins/overview)) and theme ([themes documentation](https://docs.ctfd.io/docs/themes/overview)) interfaces for customization.
*   **Data Management:**
    *   Supports importing and exporting CTF data for archival.
*   **And Much More!**

## Getting Started

### Installation

1.  **Install Dependencies:**  `pip install -r requirements.txt`
    *   You can use `prepare.sh` for system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to your needs.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

### Deployment Options

*   **Docker:**
    *   Run the pre-built image: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Use Docker Compose:  `docker compose up` (from the source repository)
*   **Documentation:** Refer to the official [CTFd documentation](https://docs.ctfd.io/) for in-depth [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd in action:  [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** For professional support or custom projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments without the infrastructure management, check out the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker.  MLC provides event scheduling, team tracking, and single sign-on.

To integrate:

1.  Register for an account on MLC.
2.  Create an event.
3.  Enter the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
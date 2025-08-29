# CTFd: The Premier Capture The Flag Framework

CTFd is a powerful and customizable framework that empowers you to easily create and manage engaging Capture The Flag (CTF) competitions.  [Explore the original repository](https://github.com/CTFd/CTFd) for full details.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd offers a comprehensive set of features to facilitate all aspects of CTF competitions:

*   **Challenge Creation & Management:**
    *   Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Support for dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Extensible challenge plugin architecture.
    *   Support for static and regex-based flags, along with custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads support using the server or an S3-compatible backend.
    *   Challenge attempt limits and challenge hiding.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allow users to play solo or collaborate in teams.
*   **Scoreboard & Ranking:**
    *   Automatic tie resolution for fair rankings.
    *   Option to hide scores from the public.
    *   Scoreboard freezing for added suspense.
    *   Scoregraphs comparing top teams and individual team progress graphs.
*   **Content & Communication:**
    *   Markdown-powered content management system.
    *   SMTP and Mailgun email support with email confirmation and password reset.
*   **Event Automation:**
    *   Automatic competition start and end times.
*   **Team Management:**
    *   Team management, hiding, and banning capabilities.
*   **Customization & Extensibility:**
    *   Extensive customization through plugin ([plugin docs](https://docs.ctfd.io/docs/plugins/overview)) and theme ([theme docs](https://docs.ctfd.io/docs/themes/overview)) interfaces.
*   **Data Management:**
    *   Import and export CTF data for easy archiving and sharing.
*   **And More!**

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Docker:**

Pre-built Docker images are available: `docker run -p 8000:8000 -it ctfd/ctfd`

Alternatively, use Docker Compose: `docker compose up` (from the source repository).

Refer to the [CTFd Documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments without infrastructure management, visit [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is heavily integrated with [MajorLeagueCyber](https://majorleaguecyber.org/).  MLC provides CTF event scheduling, team tracking, and single sign-on capabilities. Register your CTF with MajorLeagueCyber to enable automatic user login, scoring, writeup submission, and event notifications.

To integrate, register an account, create an event, and add the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
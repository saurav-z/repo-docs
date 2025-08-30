# CTFd: The Open-Source Capture The Flag Platform

**CTFd is a powerful and customizable platform designed to host and manage your own Capture The Flag (CTF) competitions with ease.** ([See the original repository](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a comprehensive set of features to create engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags via an admin interface.
    *   Support for dynamic scoring, unlockable challenges, and challenge plugins.
    *   Static and Regex-based flags, with custom flag plugins.
    *   Unlockable hints to guide participants.
    *   File uploads to the server or an S3-compatible backend.
    *   Challenge attempt limits and hiding challenges for better control.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or collaborate in teams.
*   **Scoreboard & Reporting:**
    *   Automatic tie resolution in the scoreboard.
    *   Option to hide scores from the public or freeze them at a specific time.
    *   Scoregraphs to visualize team progress and compare top teams.
*   **Content & Communication:**
    *   Markdown content management system for rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support, including confirmation and password reset features.
*   **Administration & Customization:**
    *   Automated competition start and end times.
    *   Team management tools, including hiding and banning.
    *   Extensive customization options through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Import and export CTF data for archival and portability.
*   **Additional Features:**
    *   Comprehensive feature set designed for a great CTF experience.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt` or use the `prepare.sh` script.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

**Docker:** Use pre-built images:  `docker run -p 8000:8000 -it ctfd/ctfd` or use Docker Compose:  `docker compose up`

**Resources:**

*   [CTFd Documentation](https://docs.ctfd.io/)
*   [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)

## Live Demo

Experience CTFd firsthand:  https://demo.ctfd.io/

## Support

For basic support and community interaction, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support and custom projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in a managed CTFd deployment? Check out [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## Integration with MajorLeagueCyber

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker.  By registering your CTF event with MajorLeagueCyber, users can enjoy automatic login, team and individual score tracking, writeup submission, and event notifications.

To integrate, register an account, create an event, and install the client ID and client secret in your `CTFd/config.py` or admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
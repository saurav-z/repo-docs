# CTFd: The Premier Capture The Flag (CTF) Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and customizable open-source platform designed to create and manage your own Capture The Flag (CTF) competitions with ease.** ([View the original repository](https://github.com/CTFd/CTFd))

## Key Features

CTFd offers a robust set of features to build engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges for added complexity.
    *   Implement unlockable challenges for progressive difficulty.
    *   Utilize a flexible challenge plugin architecture to build custom challenges.
    *   Support static & regex based flags
    *   Create custom flag plugins.
    *   Offer unlockable hints to guide players.
    *   Enable file uploads to the server or S3-compatible backends.
    *   Limit challenge attempts and hide challenges for strategic gameplay.
*   **Competition Modes:**
    *   Run individual or team-based competitions to suit your event's goals.
*   **Scoring and Leaderboards:**
    *   Automated tie resolution on the scoreboard.
    *   Option to hide scores from the public for added suspense.
    *   Freeze scores at a specific time to preserve the final standings.
    *   Scoregraphs comparing the top 10 teams and team progress graphs
*   **Content Management & Communication:**
    *   Utilize a Markdown content management system for rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support for notifications and password recovery.
    *   Email confirmation support
    *   Implement forgot password support
*   **Event Management:**
    *   Schedule automatic competition start and end times.
    *   Manage teams, including hiding and banning options.
*   **Customization:**
    *   Customize everything with plugins ([plugin documentation](https://docs.ctfd.io/docs/plugins/overview)) and themes ([theme documentation](https://docs.ctfd.io/docs/themes/overview)).
*   **Data Management:**
    *   Import and export CTF data for archiving or reuse.
*   **And Much More!**

## Installation

1.  **Install dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF.
3.  **Run:** Use `python serve.py` or `flask run` to start the server.

**Docker:**

*   Run the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose:  `docker compose up` (from the source repository)

Refer to the [CTFd docs](https://docs.ctfd.io/) for [detailed deployment](https://docs.ctfd.io/docs/deployment/installation) and [getting started](https://docs.ctfd.io/tutorials/getting-started/) guides.

## Live Demo

Explore the platform firsthand at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For community support, visit the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support and project-specific inquiries, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF setup with managed CTFd deployments.  Visit [the CTFd website](https://ctfd.io/) for more information.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker that offers event scheduling, team tracking, and single sign-on capabilities.  Register your event with MajorLeagueCyber to enable automatic user logins, score tracking, writeup submissions, and event notifications.

To integrate:

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
# CTFd: The Premier Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ci-mysql.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/lint.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and customizable open-source platform that empowers you to create and run your own Capture The Flag (CTF) competitions.**  Get started with your CTF today by exploring the [CTFd](https://github.com/CTFd/CTFd) repository!

## Key Features of CTFd

CTFd offers a robust set of features to make running a CTF easy and engaging:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Implement dynamic scoring challenges.
    *   Utilize unlockable challenge support and customizable plugins.
    *   Support for static and regex-based flags.
    *   Create custom flag plugins.
    *   Offer unlockable hints to assist players.
    *   Enable file uploads to the server or an S3-compatible backend.
    *   Control challenge attempts and hide challenges as needed.
    *   Benefit from automatic brute-force protection.

*   **Competition Modes:**
    *   Support individual and team-based competitions.
    *   Allow users to compete solo or form teams.

*   **Scoreboard & Analytics:**
    *   Automated tie resolution.
    *   Option to hide scores from the public.
    *   Freeze scores at a specific time to preserve the leader board.
    *   View score graphs comparing the top teams and team progress graphs.

*   **Content & Communication:**
    *   Utilize a markdown content management system.
    *   Integrate SMTP and Mailgun email support.
    *   Enable email confirmation and password recovery features.

*   **Event Management:**
    *   Automate competition start and end times.
    *   Manage teams, including hiding and banning functionalities.

*   **Customization & Integration:**
    *   Extensive customization options through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archival and easy migration.

*   **And so much more!**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker:**

Use the auto-generated Docker images:
`docker run -p 8000:8000 -it ctfd/ctfd`

Or use Docker Compose:
`docker compose up` (from the source repository)

Consult the [CTFd docs](https://docs.ctfd.io/) for [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide for comprehensive guidance.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support.

*   For commercial support, contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/).

## Managed Hosting

Interested in managed CTFd deployments? Visit [the CTFd website](https://ctfd.io/) to learn more.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on. Register your CTF event with MLC to allow users to login, track scores, submit writeups, and receive event notifications. To integrate, obtain your client ID and secret from MLC and configure them in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
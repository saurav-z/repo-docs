# CTFd: The Ultimate Capture The Flag (CTF) Framework

<p align="center">
  <img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="200"/>
</p>

CTFd is an open-source, easy-to-use, and highly customizable Capture The Flag framework, perfect for hosting your own cybersecurity competitions! **(View the original repository on GitHub)** ([https://github.com/CTFd/CTFd](https://github.com/CTFd/CTFd)).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ctfd_mysql_ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd empowers you to create engaging and educational CTF events with a wide range of features:

*   **Challenge Management:**
    *   Create and manage challenges with ease through the admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom challenge types via a plugin architecture.
    *   Implement static and regex-based flags.
    *   Includes support for file uploads to the server or Amazon S3-compatible backend.
    *   Control challenge attempts and hide challenges.
    *   Implement custom flag plugins and unlockable hints.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Team management features including hiding and banning.
*   **Scoreboard & Ranking:**
    *   Automatic tie resolution.
    *   Hide scores from the public.
    *   Freeze scores at a specific time.
    *   Score graphs comparing the top 10 teams and team progress graphs.
*   **Content Management:**
    *   Markdown content management system for creating rich content.
*   **Communication & Notifications:**
    *   SMTP and Mailgun email support.
    *   Email confirmation and password reset features.
*   **Automation & Customization:**
    *   Automated competition start and end times.
    *   Highly customizable through plugin and theme interfaces. ([Plugin Documentation](https://docs.ctfd.io/docs/plugins/overview), [Theme Documentation](https://docs.ctfd.io/docs/themes/overview))
*   **Data Management:**
    *   Importing and exporting of CTF data for archiving and backups.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker Quickstart:**

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

**Docker Compose:**
Use Docker Compose with the following command from the source repository:

```bash
docker compose up
```

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for assistance and discussions.

*   **Commercial Support:**  For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in managed CTFd deployments? Check out [the CTFd website](https://ctfd.io/) to learn more.

## Integration with MajorLeagueCyber (MLC)

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.  Integrating with MLC allows users to easily log in, track scores, submit writeups, and receive event notifications.

To integrate with MajorLeagueCyber:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Input the client ID and client secret in `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
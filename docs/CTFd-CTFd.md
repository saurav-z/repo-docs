# CTFd: The Premier Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ctfd_mysql_ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and user-friendly framework that empowers you to create and manage your own Capture The Flag (CTF) competitions with ease.** ([View the original repository](https://github.com/CTFd/CTFd))

## Key Features of CTFd

CTFd is designed to be a versatile platform for CTF organizers. Here's what makes it stand out:

*   **Intuitive Admin Interface:**
    *   Create custom challenges, categories, hints, and flags.
    *   Support for dynamic scoring challenges to keep things interesting.
    *   Unlockable challenge support.
    *   Challenge plugin architecture for custom challenge development.
    *   Static & Regex-based flag types for flexibility.
    *   Custom flag plugins for advanced scenarios.
    *   Unlockable hints to guide participants.
    *   File uploads to the server or Amazon S3-compatible backends.
    *   Limit challenge attempts to prevent brute-forcing.
    *   Hide challenges until ready.
*   **Team and Individual Competitions:**
    *   Support for both individual and team-based competitions.
    *   Allows users to compete solo or form teams.
*   **Robust Scoring and Leaderboard:**
    *   Automated scoreboard with tie resolution.
    *   Option to hide scores from the public to increase suspense.
    *   Score freezing at specific times for exciting reveals.
*   **Advanced Features:**
    *   Scoregraphs for comparing team progress.
    *   Markdown content management system for rich challenge descriptions.
    *   SMTP & Mailgun email integration for notifications and password recovery.
    *   Automated competition start and end times.
    *   Team management features: hiding, banning, etc.
    *   Customize everything with plugins and themes.
    *   Import and export CTF data for archiving and portability.

## Getting Started with CTFd

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your needs.
3.  **Run:** Use `python serve.py` or `flask run` to run the application in debug mode.

**Docker Deployment:**

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

**Docker Compose:**

Run `docker compose up` from the source repository.

Detailed deployment instructions are available in the [CTFd documentation](https://docs.ctfd.io/docs/deployment/installation).  Refer to the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide for more information.

## Live Demo

Experience CTFd firsthand at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

Get help and connect with other CTFd users:

*   **Community Forum:** [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

## Managed Hosting & Commercial Support

If you'd like to avoid managing your own infrastructure, consider a managed CTFd deployment, or if you need commercial support, visit [the CTFd website](https://ctfd.io/contact/).

## Integration with MajorLeagueCyber

CTFd integrates seamlessly with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker.  MLC provides:

*   Event scheduling.
*   Team tracking.
*   Single sign-on for events.

To integrate CTFd with MLC:

1.  Register for an account on MajorLeagueCyber.
2.  Create an event.
3.  Insert the client ID and secret into `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
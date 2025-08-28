# CTFd: The Ultimate Capture The Flag (CTF) Platform

CTFd is a powerful and user-friendly platform designed to help you create and manage engaging Capture The Flag competitions.  [Check out the original repository](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd provides a robust set of features to make your CTF experience seamless and customizable:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags easily through the admin interface.
    *   Supports dynamic scoring and unlockable challenges for added complexity.
    *   Utilize a challenge plugin architecture to create custom challenges.
    *   Choose from static and regex-based flags.
    *   Offer unlockable hints to guide participants.
    *   Enable file uploads to your server or an Amazon S3-compatible backend.
    *   Set challenge attempt limits and hide challenges.
    *   Automatic brute-force protection to secure your CTF.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Teams can play together, fostering collaboration.
*   **Scoreboard & Tracking:**
    *   Automatic tie resolution on the scoreboard.
    *   Option to hide scores from the public.
    *   Freeze scores at a specific time for exciting finales.
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system for rich content.
    *   SMTP and Mailgun email support for notifications, confirmations, and password resets.
    *   Automated competition start and end times.
*   **Customization & Administration:**
    *   Extensive customization through the plugin and theme interfaces.
    *   Team management features, including hiding and banning.
    *   Import and export CTF data for backups and archival.
*   **And More:** Additional features to provide a comprehensive CTF experience.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

**Docker:**

Use the auto-generated Docker image with:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or use Docker Compose from the source repository:

```bash
docker compose up
```

For more detailed instructions and deployment options, consult the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

[Explore a live demo of CTFd](https://demo.ctfd.io/) to experience the platform firsthand.

## Support & Community

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support and discussions.
*   For commercial support and special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

[Check out the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd is tightly integrated with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on.  Registering your CTF with MLC allows for automatic user logins, score tracking, writeup submissions, and event notifications.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
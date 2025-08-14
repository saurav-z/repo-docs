# CTFd: The Open-Source Capture The Flag Platform

CTFd is a versatile and user-friendly platform designed to help you easily create and run your own Capture The Flag (CTF) competitions. ([See the original repository](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd offers a robust set of features to create engaging and challenging CTF experiences:

*   **Easy Challenge Creation:** Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Dynamic scoring challenges
    *   Unlockable challenge support
    *   Challenge plugin architecture
    *   Static & Regex based flags
    *   Custom flag plugins
    *   Unlockable hints
    *   File uploads and Amazon S3-compatible backend support
    *   Limit challenge attempts & hide challenges
    *   Automatic bruteforce protection
*   **Flexible Competition Modes:** Supports both individual and team-based competitions.
    *   Allow users to play alone or in teams.
*   **Comprehensive Scoreboard:** Features an automatic tie resolution scoreboard with options for:
    *   Hiding scores from public view.
    *   Freezing scores at a specific time.
*   **Interactive Visualization:** Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management:** Markdown content management system for rich challenge descriptions and announcements.
*   **Communication Tools:** SMTP and Mailgun email support for user notifications, including confirmation and password reset functionality.
*   **Competition Management:** Automatic competition starting and ending features.
*   **User and Team Management:** Team management, hiding, and banning capabilities.
*   **Customization:** Highly customizable using the plugin and theme interfaces to tailor the platform to your needs.
*   **Data Handling:** Importing and exporting CTF data for backups and archival.
*   **And More:** Numerous additional features to create the perfect CTF environment.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF instance.
3.  **Run:** Start the server with `python serve.py` or `flask run` for debug mode.

**Docker:**

*   **Run pre-built image:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:**  Use `docker compose up` from the source repository.

For detailed deployment options and a Getting Started guide, refer to the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For community support and discussions, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or project-specific needs, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Explore managed CTFd deployments on [the CTFd website](https://ctfd.io/) if you prefer to avoid infrastructure management.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on. Register your CTF event to enable automatic user login, score tracking, write-up submissions, and event notifications.  To integrate, simply add the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
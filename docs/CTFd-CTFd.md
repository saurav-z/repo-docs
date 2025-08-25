# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the premier open-source Capture The Flag (CTF) platform, empowering you to easily create, manage, and run engaging cybersecurity competitions.** ([See the original repository](https://github.com/CTFd/CTFd))

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/mysql-ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Easy Challenge Creation:** Design and manage challenges, categories, hints, and flags directly through the admin interface.
    *   Dynamic Scoring Challenges
    *   Unlockable Challenge Support
    *   Challenge Plugin Architecture for custom challenge types
    *   Static & Regex-Based Flags
    *   Custom Flag Plugins
    *   Unlockable Hints
    *   File Uploads (server or S3 compatible backend)
    *   Challenge Attempt Limits & Challenge Hiding
    *   Automatic Brute-Force Protection
*   **Flexible Competition Modes:** Support for both individual and team-based competitions.
*   **Comprehensive Scoreboard:**
    *   Automatic Tie Resolution
    *   Option to Hide Scores
    *   Score Freezing at specific times
    *   Scoregraphs (Top 10 teams and team progress)
*   **Content Management:** Built-in Markdown editor for creating rich content.
*   **Email Integration:** SMTP and Mailgun support for email notifications, confirmation, and password resets.
*   **Competition Control:** Automated competition starting and ending times.
*   **Team Management:** Team management, hiding, and banning features.
*   **Extensive Customization:**  Highly customizable with plugin and theme interfaces.
*   **Data Management:** Import and export CTF data for backups and archiving.
*   **And much more!**

## Getting Started with CTFd:

1.  **Install Dependencies:** `pip install -r requirements.txt` or use `prepare.sh` to install system dependencies.
2.  **Configure CTFd:** Modify `CTFd/config.ini` to match your desired settings.
3.  **Run the Application:** Use `python serve.py` or `flask run` for debug mode.

**Docker Deployment:**
*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   `docker compose up`

For detailed deployment options, consult the [CTFd docs](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd in action: [Live Demo](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the vibrant [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for help and discussions.
*   **Commercial Support:** Contact us for commercial support and custom projects: [Contact us](https://ctfd.io/contact/).

## Managed CTFd Hosting

Looking for managed hosting to eliminate infrastructure management? Visit the [CTFd website](https://ctfd.io/) for managed deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker that provides event scheduling, team tracking, and single sign-on.  Integrating your CTF with MLC allows users to automatically log in, track scores, submit writeups, and receive notifications.
To integrate, register an account, create an event, and enter the client ID and secret in `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
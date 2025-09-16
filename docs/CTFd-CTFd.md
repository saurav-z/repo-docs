# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the go-to open-source platform for creating and hosting engaging Capture The Flag (CTF) competitions.**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## What is CTFd?

CTFd is a versatile and customizable Capture The Flag (CTF) platform designed for ease of use. Whether you're organizing a cybersecurity training event, a university competition, or a fun weekend challenge, CTFd provides everything you need to create a dynamic and engaging CTF experience. Its plugin and theme architecture allows for extensive customization to fit your specific needs.

## Key Features

*   **Intuitive Admin Interface:** Easily create and manage challenges, categories, hints, and flags.
    *   Dynamic scoring challenges
    *   Unlockable challenge support
    *   Challenge plugin architecture for custom challenges
    *   Static & Regex based flags
        *   Custom flag plugins
    *   Unlockable hints
    *   File uploads to server or S3 compatible backend
    *   Limit challenge attempts & hide challenges
    *   Automatic bruteforce protection
*   **Team and Individual Play:** Supports both individual and team-based competitions, fostering collaboration and friendly competition.
*   **Comprehensive Scoreboard:** Real-time scoreboard with automatic tie resolution and the option to hide or freeze scores.
    *   Hide Scores from the public
    *   Freeze Scores at a specific time
*   **Visualizations:** Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management:** Markdown-based content management system for creating rich, informative pages.
*   **Communication Tools:** SMTP and Mailgun email support for notifications and password recovery.
    *   Email confirmation support
    *   Forgot password support
*   **Competition Management:** Automatic start and end times for your CTF.
*   **User and Team Management:** Tools for team management, hiding, and banning.
*   **Customization:** Extensive customization options through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Management:** Import and export CTF data for archiving and backups.
*   **And more:** With a wide array of features and ongoing development, CTFd provides a powerful and flexible platform for your CTF needs.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your desired settings.
3.  **Run:** Start the server with `python serve.py` or `flask run` in a terminal.

**Docker:**

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

**Docker Compose:**

```bash
docker compose up
```

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

Get help and connect with other CTFd users:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, visit the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is seamlessly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for event scheduling, team tracking, and single sign-on. Integrate with MLC by registering an account, creating an event, and installing the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

[Visit the CTFd Repository on GitHub](https://github.com/CTFd/CTFd)
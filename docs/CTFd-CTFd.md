# CTFd: The Open-Source Capture The Flag Framework

**CTFd is a powerful and user-friendly open-source framework that allows you to easily create and host your own Capture The Flag (CTF) competitions.**  This versatile platform provides everything you need to engage cybersecurity enthusiasts and test their skills.  [View the original repository](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a comprehensive set of features to manage and run engaging CTF competitions, including:

*   **Challenge Creation & Management:**
    *   Admin interface for creating and customizing challenges, categories, hints, and flags.
    *   Supports dynamic scoring for challenges.
    *   Unlockable challenge support.
    *   Challenge plugin architecture for custom challenge types.
    *   Static & Regex-based flag support.
    *   Custom flag plugins.
    *   Unlockable hints to guide players.
    *   File uploads to server or S3-compatible backends.
    *   Challenge attempt limits and challenge hiding.
*   **Competition Modes:**
    *   Individual and team-based competitions.
    *   Team management: hiding and banning.
*   **Scoring & Leaderboards:**
    *   Automatic tie resolution.
    *   Option to hide scores from the public.
    *   Score freezing at a specific time.
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system.
    *   SMTP and Mailgun email support.
    *   Email confirmation and password recovery support.
*   **Automation & Customization:**
    *   Automatic competition starting and ending.
    *   Customize everything using plugin and theme interfaces.
    *   Importing and Exporting of CTF data.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal.

**Docker Options:**

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   `docker compose up` (from source repository)

Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the community:

*   **Community Forum:** [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:** [Contact Us](https://ctfd.io/contact/) for specialized projects or commercial needs.

## Managed Hosting

For a hassle-free CTFd experience, consider managed hosting options: [CTFd website](https://ctfd.io/).

## Integration with MajorLeagueCyber

CTFd is closely integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

Register your CTF event with MajorLeagueCyber for automatic login, score tracking, writeup submissions, and event notifications.

To integrate, create an event, and add the client ID and client secret into the `CTFd/config.py` or admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo: [Laura Barbera](http://www.laurabb.com/)
*   Theme: [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound: [Terrence Martin](https://soundcloud.com/tj-martin-composer)
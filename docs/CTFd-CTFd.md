# CTFd: The Premier Capture The Flag Framework

<img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="200"/>

CTFd is a powerful and user-friendly Capture The Flag (CTF) framework designed for hosting and managing engaging cybersecurity competitions.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd offers a comprehensive suite of features to create, manage, and run dynamic CTF events:

*   **Challenge Creation & Management:**
    *   Create custom challenges with ease via an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges, and custom challenge types.
    *   Integrate with challenge plugins for custom challenge functionality.
    *   Implement static, regex-based, and custom flag plugins.
    *   Manage hints, file uploads (to server or S3-compatible backends), and challenge attempts.
    *   Hide challenges for gradual release.
*   **Competition Modes & Team Features:**
    *   Support for both individual and team-based competitions.
    *   Team management tools (hiding and banning).
*   **Scoreboard & Reporting:**
    *   Automated tie resolution.
    *   Option to hide scores.
    *   Score freezing capabilities.
    *   Scoregraphs and team progress visualizations.
*   **Content Management & Communication:**
    *   Markdown content support for rich challenge descriptions and announcements.
    *   SMTP and Mailgun email support (including confirmation and password reset).
*   **Event Control & Customization:**
    *   Automated competition start and end times.
    *   Extensive customization via [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Import and export CTF data.
*   **Additional Features:**
    *   Bruteforce protection.

## Installation

Get started with CTFd using these simple steps:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies via `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run the Application:** Use `python serve.py` or `flask run` for development/debug mode.

**Docker:**

*   Run with the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose (from the source repository): `docker compose up`

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore the functionality of CTFd with the live demo: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

Get support and connect with the CTFd community:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or specialized projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF hosting with a managed CTFd deployment: [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/) (MLC) for advanced CTF management and participant engagement. MLC provides event scheduling, team tracking, and single sign-on capabilities.  Integrate with MLC by registering an account, creating an event, and adding the client ID and secret to `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)

## Contribute

Contribute to the CTFd project on [GitHub](https://github.com/CTFd/CTFd)!
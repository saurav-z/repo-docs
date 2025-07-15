# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a flexible and user-friendly open-source platform designed for hosting Capture The Flag (CTF) competitions.**  This comprehensive framework provides everything you need to create engaging and educational cybersecurity challenges. (See the original repo: [CTFd on GitHub](https://github.com/CTFd/CTFd))

## Key Features

*   **Easy Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges.
    *   Offers unlockable challenge support.
    *   Provides a flexible challenge plugin architecture for custom challenges.
    *   Includes static and regex-based flag options.
    *   Allows custom flag plugins.
    *   Provides unlockable hints for challenges.
    *   Supports file uploads to the server or Amazon S3-compatible backends.
    *   Allows limiting challenge attempts & hiding challenges.
    *   Includes automatic brute-force protection.
*   **Competition Modes:**
    *   Supports individual and team-based competitions.
    *   Allows users to play individually or form teams.
*   **Scoreboard & Reporting:**
    *   Features a scoreboard with automatic tie resolution.
    *   Allows hiding scores from the public.
    *   Allows freezing scores at a specific time.
    *   Provides scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management & Communication:**
    *   Utilizes a Markdown content management system.
    *   Includes SMTP + Mailgun email support.
    *   Offers email confirmation and password reset support.
*   **Competition Management:**
    *   Supports automatic competition starting and ending.
    *   Provides team management features, including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable via [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Supports importing and exporting CTF data for archiving.
*   **And More:** Explore many other features to enhance your CTF experience.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify [CTFd/config.ini](https://github.com/CTFd/CTFd/blob/master/CTFd/config.ini) to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

### Deployment Options

*   **Docker:**  Use the pre-built Docker image with: `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:**  Deploy using Docker Compose with: `docker compose up` (from the source repository)

Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or special projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, check out the official [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is integrated with [MajorLeagueCyber](https://majorleaguecyber.org/).  MLC offers event scheduling, team tracking, and single sign-on.

To integrate, register an account on MajorLeagueCyber, create an event, and add your Client ID and Secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, open-source platform that makes it easy to create and run your own Capture The Flag (CTF) competitions.**  Learn more and contribute on the [original CTFd repository](https://github.com/CTFd/CTFd).

## Key Features of CTFd:

CTFd offers a comprehensive set of features to run engaging and customizable CTF events:

*   **Challenge Creation & Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges.
    *   Includes unlockable challenge support.
    *   Offers a plugin architecture for custom challenges.
    *   Supports static and regex-based flags.
    *   Offers custom flag plugins.
    *   Includes unlockable hints.
    *   Allows file uploads to server or S3-compatible backends.
    *   Provides challenge attempt limits and challenge hiding.
    *   Integrates automatic brute-force protection.
*   **Competition Structure & Scoring:**
    *   Supports both individual and team-based competitions.
    *   Automated scoring with tie resolution.
    *   Option to hide scores from the public or freeze them at a specific time.
    *   Scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & User Management:**
    *   Utilizes a Markdown content management system.
    *   Offers SMTP and Mailgun email support (including confirmation and password reset).
    *   Automated competition start and end times.
    *   Provides team management, hiding, and banning functionalities.
*   **Customization & Integration:**
    *   Highly customizable with plugin and theme interfaces.
    *   Allows importing and exporting CTF data.
    *   Seamless integration with MajorLeagueCyber for event scheduling, team tracking, and SSO.
*   **And much more:**
    *   Offers a variety of features to enhance CTF experiences.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

### Docker Installation:

**Run using Docker:**
```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

**Run using Docker Compose:**
```bash
docker compose up
```

Refer to the [CTFd documentation](https://docs.ctfd.io/) for in-depth [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd in action: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the community:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

For commercial support or specialized project needs, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd experience? Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd integrates closely with [MajorLeagueCyber](https://majorleaguecyber.org/), which offers event scheduling, team tracking, and SSO for CTF events. Integrate your CTF by registering an account and adding your client ID/secret to your `CTFd/config.py` or admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
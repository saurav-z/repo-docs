# CTFd: The Ultimate Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly and highly customizable Capture The Flag (CTF) platform perfect for cybersecurity enthusiasts and educational institutions.**

## Key Features of CTFd

CTFd offers a comprehensive set of features to create and manage engaging CTF competitions:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags via an intuitive admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom challenge plugins for maximum flexibility.
    *   Offers both static and regular expression-based flags.
    *   Includes support for unlockable hints and challenge attempt limitations.
    *   Allows file uploads to the server or an S3-compatible backend.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Analytics:**
    *   Provides a robust scoreboard with automatic tie resolution.
    *   Offers options to hide scores and freeze scores at specific times.
    *   Includes score graphs comparing the top teams and individual team progress graphs.
*   **Content & Communication:**
    *   Integrates a Markdown content management system for announcements and challenge descriptions.
    *   Offers SMTP and Mailgun email support, including email confirmation and password reset features.
*   **Competition Control & Customization:**
    *   Provides automatic competition start and end times.
    *   Includes team management features (hiding and banning).
    *   Extensive plugin and theme interfaces to customize the CTF experience.
    *   Supports importing and exporting CTF data for archival purposes.
*   **And much more:** Explore additional features and customization options.

## Installation

Get started with CTFd by following these steps:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies via apt.
2.  **Configure CTFd:** Modify `CTFd/config.ini` to customize your CTF instance.
3.  **Run CTFd:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker Deployment:**

*   **Simple Run:**  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:**  `docker compose up` (from the source repository)

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed deployment and [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) instructions.

## Live Demo

Experience CTFd firsthand at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   Contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/) for commercial support or custom projects.

## Managed Hosting

For managed CTFd deployments without the hassle of infrastructure management, check out [the CTFd website](https://ctfd.io/).

## Integration with MajorLeagueCyber

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on. Register your CTF event with MajorLeagueCyber to enable features like automatic user login, score tracking, writeup submissions, and event notifications.

To integrate with MajorLeagueCyber, configure the `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` in `CTFd/config.py` or within the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

---

**[View the CTFd Repository on GitHub](https://github.com/CTFd/CTFd)**
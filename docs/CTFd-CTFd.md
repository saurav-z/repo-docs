# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly, highly customizable, and open-source framework designed to help you create and manage your own Capture The Flag (CTF) competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd offers a comprehensive suite of features to power your CTF events:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring, unlockable challenges, and custom challenge plugins.
    *   Offers both static and regex-based flags, with support for custom flag plugins.
    *   Includes unlockable hints, file uploads (server or S3-compatible backend), challenge attempt limits, and challenge hiding.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoring & Leaderboards:**
    *   Features a leaderboard with automatic tie resolution.
    *   Allows hiding scores from the public and score freezing.
    *   Provides score graphs to compare team progress.
*   **Content Management:**
    *   Utilizes a Markdown-based content management system.
*   **Communication & Notifications:**
    *   Offers SMTP and Mailgun email support, including email confirmation and password reset features.
    *   Provides automatic competition start and end times.
*   **User & Team Management:**
    *   Includes team management, hiding, and banning functionalities.
*   **Customization:**
    *   Extensive customization options using plugins and themes.
*   **Data Management:**
    *   Supports importing and exporting CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependency installation via apt.
2.  **Configure:** Modify `CTFd/config.ini` to your specific requirements.
3.  **Run the application:** Use `python serve.py` or `flask run` in a terminal to launch in debug mode.

**Docker:**

Use the auto-generated Docker images:
`docker run -p 8000:8000 -it ctfd/ctfd`

Or use Docker Compose:
`docker compose up` (from the source repository)

## Resources

*   [CTFd Documentation](https://docs.ctfd.io/)
*   [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)
*   [Live Demo](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** For commercial support or custom projects, contact us through the [contact page](https://ctfd.io/contact/).
*   **Managed Hosting:** Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/) for event scheduling, team tracking, and single sign-on. Register your CTF event with MLC to enable user login, score tracking, writeup submissions, and event notifications. To integrate with MLC:

1.  Register an account on MLC.
2.  Create an event.
3.  Install the client ID and client secret in the relevant section of `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
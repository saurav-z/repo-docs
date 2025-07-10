# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a user-friendly, highly customizable, and open-source Capture The Flag (CTF) platform that makes hosting CTFs a breeze.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd:

*   **Intuitive Challenge Creation:**
    *   Create challenges, categories, hints, and flags directly through the admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom flag plugins.
    *   Includes file uploads and challenge attempt limitations.
*   **Flexible Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Comprehensive Scoreboard:**
    *   Offers a scoreboard with automatic tie resolution.
    *   Allows for hiding scores or freezing them at a specific time.
    *   Includes scoregraphs and team progress graphs.
*   **Content Management:**
    *   Utilizes a Markdown-based content management system.
*   **Communication & Notification:**
    *   Provides SMTP and Mailgun email support, including email confirmation and password recovery.
    *   Offers automatic competition start and end times.
*   **Team Management:**
    *   Includes features for team management, hiding, and banning.
*   **Customization:**
    *   Highly customizable through a plugin and theme system.
*   **Data Management:**
    *   Allows importing and exporting CTF data for archival purposes.
*   **Additional Features:**
    *   Includes support for plugins that let you create custom challenges.
    *   Offers automatic brute-force protection.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt` or use the `prepare.sh` script (with `apt`).
2.  **Configure:** Modify `CTFd/config.ini` to match your preferences.
3.  **Run:** Execute `python serve.py` or `flask run` to start in debug mode.

**Docker Deployment:**
You can utilize the auto-generated Docker images using:
`docker run -p 8000:8000 -it ctfd/ctfd`

Or using Docker Compose:
`docker compose up` (from source repository)

For detailed deployment instructions, consult the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

Get basic support through the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

For commercial support or specialized projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments, visit the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker. MLC provides event scheduling, team tracking, and single sign-on functionality.

Integrate with MajorLeagueCyber by:
1. Registering an account.
2. Creating an event.
3. Adding the client ID and secret to the appropriate section in `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
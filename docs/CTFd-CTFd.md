# CTFd: The Customizable Capture The Flag Framework

**CTFd is an open-source, user-friendly framework designed to help you easily create and manage Capture The Flag (CTF) competitions.**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ctfd-mysql-ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/lint.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

CTFd provides a robust and flexible platform for running engaging CTF events, including:

*   **Easy Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges for added complexity.
    *   Implement unlockable challenge support.
    *   Utilize a challenge plugin architecture for custom challenge types.
    *   Support static & regex-based flags with custom flag plugins.
    *   Implement unlockable hints to guide participants.
    *   Support file uploads to your server or an Amazon S3-compatible backend.
    *   Set challenge attempt limits and hide challenges.
    *   Automatic bruteforce protection.
*   **Team & Individual Competitions:**
    *   Allows for both individual and team-based competitions.
    *   Supports team formation.
*   **Comprehensive Scoreboard:**
    *   Automatic tie resolution.
    *   Option to hide scores from the public.
    *   Freeze scores at specific times.
*   **Visualizations & Content Management:**
    *   Score graphs comparing top teams and team progress graphs.
    *   Markdown content management system.
*   **Communication & Automation:**
    *   SMTP + Mailgun email support.
    *   Email confirmation and password recovery support.
    *   Automatic competition start and end times.
*   **Administration & Customization:**
    *   Team management, hiding, and banning.
    *   Highly customizable via [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archiving.
*   **And Much More!**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies via apt.
2.  **Configure CTFd:** Modify `CTFd/config.ini` to your preferred settings.
3.  **Run the Application:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker:** Utilize pre-built Docker images for easy deployment:

`docker run -p 8000:8000 -it ctfd/ctfd`

**Docker Compose:**  Use the provided `docker compose up` command from the source repository.

For detailed deployment instructions and getting started guides, see the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support from the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

For commercial support, contact us at [https://ctfd.io/contact/](https://ctfd.io/contact/).

## Managed Hosting

Looking for managed CTFd deployments? Visit [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), providing event scheduling, team tracking, and single sign-on capabilities.

To integrate, register an account, create an event, and add the client ID and client secret in `CTFd/config.py` or the admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Contributing

We welcome contributions! Check out the [CTFd GitHub repository](https://github.com/CTFd/CTFd) to get involved.

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

---

**[Back to the CTFd GitHub Repository](https://github.com/CTFd/CTFd)**
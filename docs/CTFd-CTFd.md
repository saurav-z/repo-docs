# CTFd: The Open-Source Capture The Flag Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, open-source platform designed to help you host and manage your own Capture The Flag (CTF) competitions with ease.**  ([View the original repository](https://github.com/CTFd/CTFd))

## Key Features

CTFd provides a comprehensive set of features to create engaging and educational CTF events:

*   **Intuitive Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through an easy-to-use admin interface.
    *   Supports dynamic scoring challenges for added complexity.
    *   Includes unlockable challenge support to control progression.
    *   Challenge plugin architecture for building your own custom challenge types.
    *   Supports static and regex-based flags for diverse challenge types.
    *   Custom flag plugins available for advanced flag validation.
    *   Offers unlockable hints to guide participants.
    *   Allows file uploads to the server or integrated Amazon S3-compatible storage.
    *   Provides options to limit challenge attempts and hide challenges.
    *   Automatic brute-force protection to secure your CTF.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams to collaborate.
*   **Comprehensive Scoring & Leaderboards:**
    *   Features a dynamic scoreboard with automatic tie resolution.
    *   Offers the ability to hide scores from the public to maintain suspense.
    *   Provides options to freeze scores at specific times.
    *   Generates scoregraphs comparing the top 10 teams and team progress graphs for insightful analysis.
*   **Content Management & Communication:**
    *   Includes a Markdown content management system for creating engaging content.
    *   Offers SMTP and Mailgun email support for notifications.
    *   Supports email confirmation and forgot password features for account management.
*   **Event Management:**
    *   Automates competition start and end times.
    *   Provides robust team management capabilities, including hiding and banning teams.
*   **Customization & Extensibility:**
    *   Fully customizable using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Provides import and export functionality for archiving CTF data.
*   **And Much More...**

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependency installation via apt.
2.  **Configure:** Modify `CTFd/config.ini` to suit your CTF requirements.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

### Docker

Utilize auto-generated Docker images with:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or use Docker Compose from the source repository:

```bash
docker compose up
```

For detailed deployment instructions and a getting started guide, consult the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd's capabilities through the live demo: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support and discussions.
*   **Commercial Support:** For specialized projects and commercial support, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify CTF management with managed CTFd deployments.  Visit [the CTFd website](https://ctfd.io/) for details.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.  Register your CTF event to enable features such as automated user login, score tracking, writeup submission, and event notifications.

Integrate with MajorLeagueCyber by registering an account, creating an event, and configuring the client ID and secret within `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
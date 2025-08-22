# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the premier, open-source platform to host your own Capture The Flag (CTF) competitions, offering unparalleled ease of use and customization.**  [Learn more on the original repo](https://github.com/CTFd/CTFd).

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd provides a comprehensive set of features designed to create engaging and challenging CTF experiences:

*   **Intuitive Challenge Creation:**
    *   Create challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Challenge plugin architecture for custom challenge types.
    *   Static & Regex based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads (server or S3-compatible).
    *   Challenge attempt limits and challenge hiding.
*   **Flexible Competition Modes:**
    *   Individual and team-based competitions.
    *   Team management, hiding, and banning.
*   **Interactive Scoreboard & Analysis:**
    *   Automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing.
    *   Score graphs (top 10 teams & team progress).
*   **Content Management & Communication:**
    *   Markdown-based content management.
    *   SMTP + Mailgun email support (confirmation, password reset).
    *   Automatic competition start/end scheduling.
*   **Extensive Customization:**
    *   Plugin and theme interfaces for complete control.
    *   Importing and exporting CTF data.
*   **And much more!**

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Consider using `prepare.sh` for system dependencies (apt).
2.  **Configure:**  Modify `CTFd/config.ini` to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` for debug mode.

### Deployment Options

*   **Docker:**
    *   Run a pre-built image: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Use Docker Compose: `docker compose up` (from source repository).
*   **Documentation:** For detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide, consult the CTFd documentation.

## Live Demo

Explore CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Get basic support via the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** Contact us for commercial support or custom projects:  [https://ctfd.io/contact/](https://ctfd.io/contact/)

## Managed Hosting

Simplify your CTF setup with managed deployments: [the CTFd website](https://ctfd.io/)

## Integration with MajorLeagueCyber

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

*   **Benefits:** Automatic user login, individual/team score tracking, writeup submission, event notifications.
*   **Integration:**
    1.  Register at MajorLeagueCyber.
    2.  Create an event.
    3.  Enter the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
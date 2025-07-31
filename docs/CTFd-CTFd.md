# CTFd: The Customizable Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/ctfd-mysql-ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is the ultimate open-source platform to host and manage your own Capture The Flag (CTF) competitions.**  Offering a flexible and user-friendly experience, CTFd allows you to create engaging cybersecurity challenges for individual players or teams.

[Visit the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd:

*   **Challenge Management:**
    *   Create and manage challenges with ease through an intuitive admin interface.
    *   Supports dynamic scoring for varied difficulty levels.
    *   Implement unlockable challenges to guide participants.
    *   Utilize a powerful plugin architecture for custom challenge types.
    *   Supports static and regex-based flags.
    *   Integrate custom flag plugins.
    *   Implement unlockable hints to help players.
    *   Allow file uploads to the server or S3-compatible backends.
    *   Limit challenge attempts and hide challenges.
*   **Competition Modes:**
    *   Individual and team-based competition options.
    *   Allows players to compete solo or collaborate in teams.
*   **Scoring & Leaderboards:**
    *   Real-time scoreboards with automatic tie resolution.
    *   Option to hide scores from public view.
    *   Freeze scores at a specific time to prevent late game advantages.
*   **Visualization & Communication:**
    *   Scoregraphs for visual team comparison (top 10 teams).
    *   Team progress graphs for insight.
    *   Markdown-based content management system for rich content.
*   **Notifications & Email:**
    *   SMTP and Mailgun email support for notifications.
    *   Email confirmation for user accounts.
    *   Forgot password support.
*   **Competition Control:**
    *   Automatic competition start and end times.
    *   Team management features: hiding and banning.
*   **Customization:**
    *   Highly customizable through a robust plugin and theme system.
    *   Import and export CTF data for archiving and reuse.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to set up your CTF settings.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker:**

*   **Run with Docker (Simple):** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run with Docker Compose:**  `docker compose up` (from the source repository)

**Documentation:**

*   Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and a [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

[https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   Get help on the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

## Managed Hosting & Commercial Support

*   For managed CTFd deployments, visit [the CTFd website](https://ctfd.io/).
*   Contact us for commercial support via the [CTFd contact page](https://ctfd.io/contact/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker, providing event scheduling, team tracking, and single sign-on. This allows for:

*   Automated user logins.
*   Individual and team score tracking.
*   Writeup submissions.
*   Event notifications.

**Integration Steps:**

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret in either:
    *   `CTFd/config.py` (using `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET`).
    *   Or the admin panel.

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
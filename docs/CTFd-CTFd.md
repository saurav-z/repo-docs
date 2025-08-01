# CTFd: The Premier Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

**CTFd is an open-source Capture The Flag (CTF) platform designed for ease of use, customizability, and a fantastic user experience, making it the perfect tool for hosting engaging cybersecurity competitions.** ([Check out the original repository](https://github.com/CTFd/CTFd))

<p align="center">
  <img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="200"/>
  <img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true" alt="CTFd Scoreboard" width="400"/>
</p>

## Key Features

CTFd offers a comprehensive set of features to create and manage your CTF:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags.
    *   Supports dynamic scoring challenges.
    *   Implement unlockable challenges and hints.
    *   Plugin architecture for custom challenge types.
    *   Static and regex-based flags.
    *   Custom flag plugins.
    *   File uploads to the server or Amazon S3.
    *   Challenge attempt limits and hiding options.
    *   Automatic brute-force protection
*   **Competition Modes:**
    *   Individual and team-based competitions.
*   **Scoring & Leaderboards:**
    *   Scoreboard with automatic tie resolution.
    *   Option to hide scores.
    *   Score freezing at a specific time.
    *   Scoregraphs and team progress graphs.
*   **Content & Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support.
    *   Email confirmation and password recovery.
    *   Automatic competition start and end times.
*   **Team Management:**
    *   Team management, hiding, and banning.
*   **Customization:**
    *   Highly customizable using plugins and themes (detailed documentation available).
*   **Data Management:**
    *   Import and export CTF data for archiving.
*   **And much more!**

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker:**

*   **Run the default Docker image:**
    ```bash
    docker run -p 8000:8000 -it ctfd/ctfd
    ```
*   **Docker Compose:**
    ```bash
    docker compose up
    ```
    (From the source repository)

For detailed deployment instructions and a getting started guide, refer to the [CTFd Documentation](https://docs.ctfd.io/) and [Getting Started](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Experience CTFd firsthand at the [Live Demo](https://demo.ctfd.io/).

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us through [CTFd website](https://ctfd.io/contact/) for commercial support or custom projects.

## Managed Hosting

Looking for hassle-free CTF hosting? Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker that provides event scheduling, team tracking, and single sign-on capabilities. Register your CTF event to leverage automatic logins, score tracking, writeup submission, and important event notifications.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret into `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
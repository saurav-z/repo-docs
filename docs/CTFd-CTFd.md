# CTFd: The Ultimate Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

CTFd is a flexible and user-friendly Capture The Flag (CTF) platform designed to empower cybersecurity enthusiasts and educators.  You can find the original repo [here](https://github.com/CTFd/CTFd).

## Key Features

CTFd offers a comprehensive suite of features to create and manage engaging CTF competitions, including:

*   **Easy Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring for challenges.
    *   Offers unlockable challenge support.
    *   Utilizes a plugin architecture for custom challenge development.
    *   Supports static and regex-based flags.
    *   Allows custom flag plugins.
    *   Provides unlockable hints.
    *   Enables file uploads to the server or an Amazon S3-compatible backend.
    *   Allows limiting challenge attempts & hiding challenges.
    *   Offers automatic brute-force protection.
*   **Competition Management:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams.
    *   Includes a scoreboard with automatic tie resolution.
    *   Provides options to hide scores from the public.
    *   Allows score freezing at specific times.
*   **Visual & Interactive Elements:**
    *   Offers scoregraphs comparing the top 10 teams and team progress graphs.
    *   Utilizes a Markdown content management system.
*   **Communication & Automation:**
    *   Includes SMTP + Mailgun email support.
    *   Supports email confirmation and password recovery.
    *   Automates competition starting and ending.
*   **Administration & Customization:**
    *   Provides robust team management features (hiding, banning).
    *   Enables extensive customization through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Allows importing and exporting CTF data for archival purposes.
    *   Offers a wide array of additional features.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

    **Docker:**

    *   Run pre-built Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Use Docker Compose (from the source repository): `docker compose up`

    Refer to the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for support and discussions.
*   **Commercial Support:** For commercial support or special project inquiries, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Interested in using CTFd without managing infrastructure? Explore managed CTFd deployments at [the CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.

**Integration Steps:**

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret within `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
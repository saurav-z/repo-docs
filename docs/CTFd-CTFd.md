# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the ultimate open-source platform for hosting Capture The Flag (CTF) competitions, designed for ease of use and customization.**

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd offers a robust set of features to create and manage engaging CTF events:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges.
    *   Includes unlockable challenge support.
    *   Uses a challenge plugin architecture for custom challenges.
    *   Offers static and regex-based flags.
    *   Supports custom flag plugins.
    *   Provides unlockable hints.
    *   Allows file uploads to the server or an Amazon S3-compatible backend.
    *   Includes features to limit challenge attempts and hide challenges.
    *   Provides automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams.
*   **Scoreboard & Scoring:**
    *   Features a scoreboard with automatic tie resolution.
    *   Allows hiding scores from the public.
    *   Offers options to freeze scores at a specific time.
    *   Provides scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management:**
    *   Includes a Markdown content management system.
*   **Communication & Notifications:**
    *   Supports SMTP and Mailgun email integration.
    *   Includes email confirmation and password reset functionality.
*   **Competition Control:**
    *   Provides automated competition start and end times.
    *   Offers team management features (hiding and banning).
*   **Customization & Extensibility:**
    *   Extensive customization through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Supports importing and exporting CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script (requires apt) to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to match your needs.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

**Docker:**

*   Run the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose: `docker compose up` (from the source repository).

Refer to the [CTFd documentation](https://docs.ctfd.io/) for [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and connect with the community:

*   **Community Forum:** [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:** Contact us for specialized projects: [Contact](https://ctfd.io/contact/)

## Managed Hosting

Simplify your CTF setup with managed CTFd deployments: [CTFd Website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Add the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)

**[Back to the original repository](https://github.com/CTFd/CTFd)**
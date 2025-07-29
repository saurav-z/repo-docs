# CTFd: The Open-Source Capture The Flag Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a flexible and user-friendly platform that empowers you to create and run your own Capture The Flag (CTF) competitions.** Check out the original repository [here](https://github.com/CTFd/CTFd).

## Key Features

CTFd offers a comprehensive set of features to make running your CTF a breeze:

*   **Challenge Creation and Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges.
    *   Includes unlockable challenge support.
    *   Offers a challenge plugin architecture for custom challenge types.
    *   Supports static and regex-based flags.
    *   Includes custom flag plugins.
    *   Provides unlockable hints.
    *   Supports file uploads to the server or S3-compatible backends.
    *   Allows you to limit challenge attempts and hide challenges.
    *   Offers automatic brute-force protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams.
*   **Scoreboard and Reporting:**
    *   Provides a scoreboard with automatic tie resolution.
    *   Offers options to hide scores from the public or freeze scores at a specific time.
    *   Generates scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content and Communication:**
    *   Uses a Markdown content management system.
    *   Supports SMTP and Mailgun email for notifications, including confirmation and password reset.
*   **Automation and Customization:**
    *   Offers automatic competition starting and ending.
    *   Provides team management, hiding, and banning features.
    *   Allows extensive customization through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Supports importing and exporting CTF data for archiving.

## Getting Started

1.  **Install Dependencies:**  `pip install -r requirements.txt`
    *   You can optionally use the `prepare.sh` script to install system dependencies via `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to enter debug mode.

## Deployment Options

*   **Docker:**
    *   Use the auto-generated Docker images with: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Or use Docker Compose: `docker compose up` (from the source repository)
*   **For more information:** Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [installation](https://docs.ctfd.io/docs/deployment/installation) instructions and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd in action at: https://demo.ctfd.io/

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us via [the contact form](https://ctfd.io/contact/) if you have a specific project that requires commercial support.

## Managed Hosting

Consider using [the CTFd website](https://ctfd.io/) for managed CTFd deployments if you want to avoid the complexity of infrastructure management.

## MajorLeagueCyber Integration

CTFd is tightly integrated with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker that offers event scheduling, team tracking, and single sign-on.  By registering your CTF event with MLC, users can seamlessly log in, track scores, submit writeups, and receive notifications.

*   **Integration steps:** Register an account on MLC, create an event, and configure the client ID and client secret in either `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
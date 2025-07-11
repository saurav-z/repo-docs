# CTFd: The Premier Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a versatile and user-friendly Capture The Flag (CTF) platform designed to streamline the creation, management, and execution of cybersecurity competitions.**

[Visit the CTFd GitHub Repository](https://github.com/CTFd/CTFd)

## Key Features

CTFd offers a comprehensive suite of features to create and manage your own CTFs:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags through the Admin Interface.
    *   Supports Dynamic Scoring Challenges.
    *   Implement Unlockable Challenge support.
    *   Leverage a Challenge plugin architecture for custom challenge types.
    *   Use Static and Regex based flags.
    *   Create Custom flag plugins.
    *   Offer Unlockable hints.
    *   Allow file uploads to the server or integrate with an Amazon S3-compatible backend.
    *   Limit challenge attempts & hide challenges.
    *   Offers Automatic bruteforce protection.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or collaborate in teams.
*   **Scoreboard & Analytics:**
    *   Features a robust Scoreboard with automatic tie resolution.
    *   Option to hide scores from the public.
    *   Freeze scores at a specific time.
    *   Provides Scoregraphs to compare top teams and track team progress.
*   **Content & Communication:**
    *   Uses a Markdown content management system.
    *   Offers SMTP + Mailgun email support.
    *   Includes email confirmation support.
    *   Features Forgot password support.
*   **Competition Control:**
    *   Supports Automatic competition starting and ending.
    *   Provides Team management, hiding, and banning features.
*   **Customization & Integration:**
    *   Customize everything using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Importing and Exporting of CTF data for archival.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

**Docker:**

*   **Run with Docker (simple):** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run with Docker Compose (recommended):** `docker compose up` (from the source repository)

For detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide, consult the CTFd documentation.

## Live Demo

Explore CTFd in action: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

Get help and connect with other CTFd users:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   For commercial support and custom projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify CTF management with managed CTFd deployments: [CTFd Website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker:

*   Provides event scheduling, team tracking, and single sign-on.
*   Register your CTF event with MajorLeagueCyber for automatic user login, score tracking, writeup submission, and event notifications.

**Integration Steps:**

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py` or in the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo: [Laura Barbera](http://www.laurabb.com/)
*   Theme: [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound: [Terrence Martin](https://soundcloud.com/tj-martin-composer)
# CTFd: The Customizable Capture The Flag Framework

CTFd is a powerful and flexible open-source platform designed to host and manage Capture The Flag (CTF) competitions.  [Check out the original repository](https://github.com/CTFd/CTFd) for the latest updates and source code.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd provides a comprehensive set of features to create engaging and customizable CTF experiences:

*   **Challenge Management:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring, unlockable challenges, and a plugin architecture for custom challenge types.
    *   Includes static and regex-based flags.
    *   Offers custom flag plugins and unlockable hints.
    *   Allows file uploads to the server or an S3-compatible backend.
    *   Provides options to limit challenge attempts and hide challenges.
    *   Automatic brute-force protection
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete individually or form teams.
*   **Scoreboard & Reporting:**
    *   Offers a scoreboard with automatic tie resolution.
    *   Ability to hide scores from the public or freeze them at a specific time.
    *   Provides score graphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Utilizes a Markdown content management system.
    *   Supports SMTP + Mailgun email for notifications (email confirmation, password reset).
    *   Allows automatic competition starting and ending times.
*   **Admin & Customization:**
    *   Team management, hiding, and banning features.
    *   Fully customizable using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Data importing and exporting for CTF archiving.

## Getting Started

To get started with CTFd:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to run in debug mode.

## Deployment Options

*   **Docker:**  Use the pre-built Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
    *   Or use Docker Compose: `docker compose up` (from the source repository directory)
*   **Documentation:** Consult the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd at:  https://demo.ctfd.io/

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, explore [managed CTFd deployments](https://ctfd.io/) on the CTFd website.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker providing event scheduling, team tracking, and single sign-on features.  Integrating CTFd with MLC allows for automatic user login, score tracking, writeup submissions, and event notifications.

To integrate:

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Enter the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
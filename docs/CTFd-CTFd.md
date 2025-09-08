# CTFd: The Open-Source Capture The Flag Framework

![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)
![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, easy-to-use, and highly customizable open-source framework designed to run your own Capture The Flag (CTF) competitions.**  [Explore the CTFd repository on GitHub](https://github.com/CTFd/CTFd).

## Key Features of CTFd

CTFd offers a comprehensive set of features to create and manage engaging CTF events:

*   **Challenge Management:**
    *   Create challenges with various types: dynamic scoring, unlockable challenges, and custom challenge plugins.
    *   Support for static & regex-based flags.
    *   Configure hints and file uploads.
    *   Limit challenge attempts and hide challenges.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Tie Resolution:**
    *   Automatic tie resolution.
    *   Option to hide scores.
    *   Scoreboard freezing.
    *   Scoregraphs to visualize team performance.
*   **Content Management:**
    *   Built-in Markdown content management system.
*   **Communication & Notifications:**
    *   SMTP and Mailgun email support.
    *   Email confirmation and password reset functionality.
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management tools (hiding and banning).
*   **Customization:**
    *   Highly customizable with [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Management:**
    *   Importing and exporting CTF data.

## Installation

1.  **Dependencies:** Install dependencies using `pip install -r requirements.txt`.
    *   Alternatively, use the `prepare.sh` script for system dependencies via `apt`.
2.  **Configuration:** Modify `CTFd/config.ini` to your preferences.
3.  **Run:** Start the server with `python serve.py` or `flask run` for debug mode.

**Docker Usage:**

*   Run a CTFd instance using the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Alternatively, use Docker Compose: `docker compose up` (from the source repository directory).

For detailed deployment instructions and a Getting Started guide, please refer to the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Explore the CTFd demo: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

Get help and connect with the CTFd community:

*   **Community Forum:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support and discussions.
*   **Commercial Support:**  For commercial support or custom project needs, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd experience?  Check out [CTFd's managed hosting options](https://ctfd.io/) for easy deployments.

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and SSO.  MLC allows users to automatically log in, track scores, submit writeups, and receive event notifications.

**Integration Steps:**

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py`:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```
    or through the admin panel.

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
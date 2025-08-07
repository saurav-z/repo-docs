# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and customizable open-source framework that makes it easy to create and manage your own Capture The Flag (CTF) competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd offers a wide range of features to create engaging and challenging CTF events:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges to adjust difficulty.
    *   Offers unlockable challenge support for a progressive challenge experience.
    *   Utilizes a flexible plugin architecture to allow custom challenge types.
    *   Supports static and regex-based flags.
    *   Offers custom flag plugins for advanced flag validation.
    *   Includes unlockable hints to guide players.
    *   Allows file uploads to the server or an Amazon S3-compatible backend.
    *   Allows to limit challenge attempts & hide challenges for a smoother gameplay.
    *   Provides automatic brute-force protection.
*   **Competition Structure:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or form teams.
*   **Scoring and Leaderboard:**
    *   Offers a real-time scoreboard with automatic tie resolution.
    *   Allows to hide scores from the public to maintain suspense.
    *   Offers options to freeze scores at a specific time.
    *   Provides scoregraphs for the top 10 teams and progress graphs for each team.
*   **Content Management & Communication:**
    *   Includes a Markdown-based content management system.
    *   Offers SMTP and Mailgun email support.
    *   Provides email confirmation and password reset support.
    *   Supports automatic competition starting and ending times.
*   **User & Team Management:**
    *   Provides team management features, including hiding and banning teams.
*   **Customization & Extensibility:**
    *   Offers extensive customization using a plugin and theme interfaces.
    *   Allows importing and exporting of CTF data for archival.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to match your desired settings.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debug mode.

### Docker

You can easily run CTFd using Docker:

*   **Run the auto-generated Docker image:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Use Docker Compose:** `docker compose up` from the source repository.

See the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Get support from the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** Contact [CTFd](https://ctfd.io/contact/) for commercial support and special projects.

## Managed Hosting

Interested in using CTFd without managing the infrastructure? Check out [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd is seamlessly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.

**Integration Steps:**

1.  Register your CTF event with MajorLeagueCyber.
2.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
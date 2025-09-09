# CTFd: The Customizable Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is the open-source platform you need to easily create and run your own Capture The Flag (CTF) competitions.** ([View on GitHub](https://github.com/CTFd/CTFd))

## Key Features of CTFd

CTFd provides a comprehensive set of features for hosting engaging and customizable CTF events:

*   **Challenge Creation and Management:**
    *   Create and manage challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges.
    *   Offers unlockable challenge support.
    *   Leverages a robust challenge plugin architecture for custom challenges.
    *   Provides static and regex-based flag options.
    *   Allows for custom flag plugins.
    *   Offers unlockable hints.
    *   Supports file uploads to the server or Amazon S3-compatible backends.
    *   Allows limiting challenge attempts and hiding challenges.
    *   Includes automatic brute-force protection.
*   **Competition Types:**
    *   Supports both individual and team-based competitions.
*   **Scoreboard & Analytics:**
    *   Provides a feature-rich scoreboard with automatic tie resolution.
    *   Allows for hiding scores from the public.
    *   Offers the ability to freeze scores at a specific time.
    *   Includes scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content & Communication:**
    *   Offers a Markdown content management system.
    *   Provides SMTP and Mailgun email support, including email confirmation and password reset functionality.
*   **Competition Control:**
    *   Enables automatic competition starting and ending.
    *   Allows for team management, including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable through the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Supports importing and exporting CTF data for archival purposes.
*   **And much more!**

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to tailor the platform to your needs.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to start in debug mode.

**Docker Options:**

*   **Pre-built image:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:**  `docker compose up` (from the source repository)
   *   For deployment options and detailed instructions, refer to the [CTFd docs](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the CTFd community:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   For commercial support or project-specific inquiries, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF hosting with managed CTFd deployments: [https://ctfd.io/](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.  Register your CTF event with MajorLeagueCyber to enable automatic user logins, score tracking, writeup submissions, and event notifications.

**Integration steps:**

1.  Register an account on MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret in the appropriate sections within `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
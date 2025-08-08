# CTFd: The Premier Capture The Flag (CTF) Platform

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

CTFd is a powerful and user-friendly framework designed to make running your own Capture The Flag (CTF) competition easy and customizable.

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd offers a comprehensive set of features to create engaging and challenging CTF events:

*   **Challenge Management:**
    *   Create and manage challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges to keep the competition exciting.
    *   Unlockable challenge support to guide players through the CTF.
    *   Extensible challenge plugin architecture for custom challenge types.
    *   Supports both static and regular expression-based flags.
    *   Custom flag plugins for advanced flag validation.
    *   Unlockable hints to provide assistance to players.
    *   File uploads to the server or integration with Amazon S3-compatible storage.
    *   Ability to limit challenge attempts and hide challenges.
    *   Automatic brute-force protection to prevent abuse.

*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to play solo or collaborate in teams.

*   **Scoring & Ranking:**
    *   Automated scoreboard with tie resolution to keep track of player progress.
    *   Option to hide scores from the public for a more exclusive experience.
    *   Score freezing feature to lock scores at a specific time for final results.
    *   Scoregraphs comparing the top 10 teams and individual team progress graphs.

*   **Content Management:**
    *   Markdown content management system for creating engaging content.

*   **Communication & Notifications:**
    *   SMTP and Mailgun email support for announcements and password resets.
    *   Email confirmation support to ensure user account verification.
    *   Forgot password support for easy account recovery.

*   **Event Management:**
    *   Automated competition starting and ending features.
    *   Team management tools including hiding and banning.

*   **Customization & Extensibility:**
    *   Extensive customization options through [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Importing and exporting of CTF data for archival purposes.

## Installation

Get started with CTFd in a few simple steps:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF settings.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal for debugging.

**Docker:**

You can quickly deploy CTFd using Docker:

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose: `docker compose up` (from the source repository)

Consult the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For basic support, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support or custom projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, consider [managed CTFd deployments](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker.  By registering your CTF event, users can enjoy:

*   Automatic login
*   Individual and team score tracking
*   Writeup submissions
*   Event notifications

**Integration Steps:**

1.  Register an account with MajorLeagueCyber.
2.  Create an event.
3.  Install the client ID and client secret in `CTFd/config.py` or the admin panel:
    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
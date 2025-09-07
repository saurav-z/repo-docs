# CTFd: The Ultimate Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions?query=workflow%3A%22CTFd+MySQL+CI%22)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions?query=workflow%3ALinting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is an open-source, easy-to-use Capture The Flag (CTF) platform perfect for running engaging cybersecurity competitions.**

[Check out the original repository](https://github.com/CTFd/CTFd)

## Key Features

CTFd provides a comprehensive suite of features to create, manage, and run exciting CTF events:

*   **Intuitive Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges and unlockable challenges.
    *   Plugin architecture enables creation of custom challenge types.
    *   Supports static and regex-based flags, and custom flag plugins.
    *   Offers unlockable hints to guide players.
    *   Allows file uploads to server or S3-compatible backends.
    *   Challenge attempt limits and challenge hiding.
*   **Flexible Competition Modes:**
    *   Supports individual and team-based competitions.
    *   Allows users to compete solo or form teams.
*   **Robust Scoreboard and Team Management:**
    *   Automatic tie resolution on the scoreboard.
    *   Option to hide scores from public view.
    *   Freeze scores at a specific time for final results.
    *   Scoregraphs comparing the top 10 teams and progress graphs.
    *   Team management, hiding, and banning.
*   **Content and Communication:**
    *   Markdown content management system.
    *   SMTP + Mailgun email support for notifications and password resets.
    *   Automatic competition starting and ending times.
*   **Extensibility and Customization:**
    *   Customize everything using plugin and theme interfaces.
    *   Importing and exporting of CTF data for archiving and backups.

## Installation and Deployment

### Prerequisites

*   Python 3.6+
*   pip

### Steps

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to configure your CTF.
3.  **Run the Application:** Use `python serve.py` or `flask run` in a terminal.

### Deployment Options

*   **Docker:** Use the auto-generated Docker image: `docker run -p 8000:8000 -it ctfd/ctfd` or `docker compose up` from the source repository.
*   **Documentation:** Consult the [CTFd docs](https://docs.ctfd.io/) for detailed deployment options and a getting started guide.

## Live Demo

Experience CTFd in action at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support and discussions.
*   **Commercial Support:** For commercial support or custom projects, contact us via the [contact page](https://ctfd.io/contact/).
*   **Managed Hosting:** Explore managed CTFd deployments on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is tightly integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker.

*   **Benefits:**
    *   Event scheduling, team tracking, and single sign-on (SSO) for events.
    *   Automated user login, score tracking, write-up submissions, and event notifications.
*   **Integration:**
    1.  Register for an account on MajorLeagueCyber.
    2.  Create a new event.
    3.  Enter the client ID and client secret within `CTFd/config.py` or the Admin Panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
# CTFd: The Customizable Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is an open-source Capture The Flag (CTF) platform that offers unparalleled flexibility and ease of use for creating and managing your own cybersecurity competitions.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features

CTFd provides a comprehensive set of features for running engaging and customizable CTFs:

*   **Challenge Creation & Management:**
    *   Create challenges with dynamic scoring.
    *   Implement unlockable challenges for progressive difficulty.
    *   Utilize a plugin architecture for custom challenge types.
    *   Support for static and regex-based flags.
    *   Create custom flag plugins.
    *   Integrate unlockable hints.
    *   Enable file uploads (to server or S3-compatible backends).
    *   Limit challenge attempts and hide challenges.
    *   Includes automatic bruteforce protection.
*   **Competition Modes:**
    *   Support individual and team-based competitions.
    *   Allow users to compete solo or collaborate in teams.
*   **Scoreboard & Reporting:**
    *   Automated tie resolution on the scoreboard.
    *   Options to hide scores from the public.
    *   Score freezing at a specific time.
    *   Score graphs comparing top teams and individual team progress.
*   **Content & Communication:**
    *   Markdown content management system for creating engaging content.
    *   SMTP and Mailgun email support.
    *   Email confirmation and password reset support.
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management tools (hiding and banning teams).
*   **Customization & Extensibility:**
    *   Extensive plugin system for customization ([plugin documentation](https://docs.ctfd.io/docs/plugins/overview)).
    *   Flexible theming capabilities ([theme documentation](https://docs.ctfd.io/docs/themes/overview)).
*   **Data Management:**
    *   Import and export CTF data for archiving and sharing.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to your specific preferences.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

### Docker

**Docker Images:**

`docker run -p 8000:8000 -it ctfd/ctfd`

**Docker Compose:**

`docker compose up` (from the source repository directory)

See the [CTFd docs](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get support and connect with the CTFd community:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or custom project assistance, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTF experience, consider a managed CTFd deployment: [CTFd website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker:

*   MLC provides event scheduling, team tracking, and single sign-on.
*   Register your CTF event with MLC for automatic user login, score tracking, writeup submission, and event notifications.
*   Integrate by adding your client ID and secret to `CTFd/config.py` or the admin panel.

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
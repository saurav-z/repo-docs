# CTFd: The Customizable Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)
[![](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true)](https://github.com/CTFd/CTFd)

**CTFd is a flexible and user-friendly Capture The Flag (CTF) platform designed to help you create engaging cybersecurity competitions with ease.**

## What is CTFd?

CTFd is a popular, open-source CTF framework built with ease of use and customizability in mind. Whether you're organizing a small training exercise or a large-scale cybersecurity competition, CTFd provides the tools you need. With a robust admin interface and plugin/theme support, CTFd can be tailored to fit your specific needs.

[![](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)](https://github.com/CTFd/CTFd)

## Key Features

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags directly from the admin interface.
    *   Support for dynamic scoring to adjust difficulty.
    *   Unlockable challenge features for progressive challenges.
    *   Extensible challenge plugin architecture for custom challenge types.
    *   Supports static and regex-based flags, with custom flag plugins.
    *   Create unlockable hints to guide participants.
    *   File upload functionality, with options for server or Amazon S3-compatible storage.
    *   Limit challenge attempts and hide challenges as needed.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or form teams.
*   **Scoreboard & Ranking:**
    *   Automated tie resolution on the scoreboard.
    *   Option to hide scores from the public.
    *   Ability to freeze scores at a specific time.
    *   Scoregraphs to compare the top 10 teams.
    *   Team progress graphs for insights.
*   **Content Management & Communication:**
    *   Built-in Markdown content management system.
    *   SMTP and Mailgun email support for announcements and notifications.
    *   Email confirmation and password reset features.
*   **Competition Control:**
    *   Automated competition start and end times.
    *   Team management features, including hiding and banning.
*   **Customization:**
    *   Highly customizable through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
*   **Data Handling:**
    *   Import and export CTF data for archiving and backups.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script for system dependencies (using apt).
2.  **Configure:** Modify `CTFd/config.ini` to customize settings.
3.  **Run:** Start the server using `python serve.py` or `flask run` in a terminal (debug mode).

**Docker:**

*   Run the auto-generated Docker image:
    `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose (from the source repository):
    `docker compose up`

**Resources:**
*   Consult the [CTFd docs](https://docs.ctfd.io/) for [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore the platform with the [live demo](https://demo.ctfd.io/).

## Support & Community

Get help from the community: [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or special projects, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Looking for a hassle-free CTFd deployment? Check out [CTFd's managed hosting options](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/) (MLC), a CTF stats tracker offering event scheduling, team tracking, and single sign-on. Register your CTF event with MLC to enable automatic login, score tracking, writeup submission, and event notifications for users.

**MLC Integration Setup:**
1.  Create an account on MajorLeagueCyber.
2.  Create a CTF event on MajorLeagueCyber.
3.  Install the client ID and client secret in `CTFd/config.py` or in the admin panel:
    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

## Contribute

Contribute to the project on [GitHub](https://github.com/CTFd/CTFd).
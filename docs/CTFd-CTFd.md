# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and user-friendly open-source framework that allows you to easily create and manage your own Capture The Flag (CTF) competitions.**

[View the original repository](https://github.com/CTFd/CTFd)

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags directly from the admin interface.
    *   Supports dynamic scoring challenges, unlockable challenges, and custom flag plugins.
    *   Includes file uploads with S3-compatible backend support.
    *   Offers options to limit challenge attempts and hide challenges.
    *   Integrates automatic brute-force protection.

*   **Competition Types & Scoring:**
    *   Supports both individual and team-based competitions.
    *   Provides a robust scoreboard with automatic tie resolution.
    *   Allows hiding scores and freezing them at specific times.
    *   Includes scoregraphs and team progress graphs.

*   **Content Management & Communication:**
    *   Features a Markdown content management system.
    *   Provides SMTP + Mailgun email support for user registration, password resets, and notifications.
    *   Offers automatic competition starting and ending options.
    *   Supports team management, hiding, and banning features.

*   **Customization & Integration:**
    *   Highly customizable using plugins and themes.
    *   Allows importing and exporting CTF data for backups.
    *   Integrates with MajorLeagueCyber for event scheduling, team tracking, and single sign-on.

## Getting Started

### Installation

1.  **Install dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to suit your preferences.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

### Deployment Options

*   **Docker:** Use the pre-built Docker image: `docker run -p 8000:8000 -it ctfd/ctfd` or Docker Compose.
*   **Docs:** Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore a live CTFd instance at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community:** Get basic support and connect with other users via the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** For commercial support or special projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For managed CTFd deployments without managing infrastructure, check out the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on. Register your CTF with MajorLeagueCyber for automatic user login and enhanced features.

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
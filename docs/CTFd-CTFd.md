# CTFd: The Customizable Capture The Flag Framework

[<img src="https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/logo.png?raw=true" alt="CTFd Logo" width="150">](https://github.com/CTFd/CTFd)

CTFd is a user-friendly and highly customizable Capture The Flag (CTF) platform, perfect for running engaging cybersecurity competitions.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

*   **Intuitive Admin Interface:** Easily create and manage challenges, categories, hints, and flags.
    *   Dynamic Scoring Challenges
    *   Unlockable challenge support
    *   Challenge plugin architecture to create custom challenges
    *   Static & Regex based flags
    *   Custom flag plugins
    *   Unlockable hints
    *   File uploads (server or S3-compatible)
    *   Challenge attempt limits & hiding
    *   Automatic brute-force protection
*   **Team & Individual Competitions:** Support for both individual and team-based gameplay.
*   **Interactive Scoreboard:** Real-time scoreboard with tie resolution, score hiding, and freeze options.
*   **Score Visualization:** Scoregraphs comparing the top teams and team progress graphs.
*   **Markdown-Based Content:** Robust content management system.
*   **Email Integration:** SMTP & Mailgun support (email confirmation, password reset).
*   **Competition Management:** Automatic start/end times and team management features (hiding, banning).
*   **Customization:** Extensive plugin and theme interfaces for complete control.
*   **Data Management:** Import and export CTF data.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to your requirements.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal.

**Docker:**

*   `docker run -p 8000:8000 -it ctfd/ctfd`
*   `docker compose up` (from source repository)

See the [CTFd documentation](https://docs.ctfd.io/) for detailed [installation](https://docs.ctfd.io/docs/deployment/installation) and [getting started](https://docs.ctfd.io/tutorials/getting-started/) guides.

## Live Demo

Explore a live demonstration of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** For commercial support or custom projects, contact us via [CTFd Website](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, explore [managed CTFd deployments](https://ctfd.io/) on the CTFd website.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and SSO.  Register your CTF event with MajorLeagueCyber for automatic user logins, score tracking, write-up submissions, and event notifications.

To integrate, create an account, create an event, and input your client ID and secret in `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)

**[Back to Top](#ctfd-the-customizable-capture-the-flag-framework)**

**[View the original repository on GitHub](https://github.com/CTFd/CTFd)**
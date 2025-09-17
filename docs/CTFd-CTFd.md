# CTFd: The Ultimate Capture The Flag Framework

**(Check out the original CTFd repository: [https://github.com/CTFd/CTFd](https://github.com/CTFd/CTFd))**

CTFd is a user-friendly and highly customizable Capture The Flag (CTF) framework designed to streamline the creation and management of your own CTF competitions.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd

CTFd boasts a comprehensive suite of features, making it ideal for running engaging and educational CTF events:

*   **Admin Interface for Easy Challenge Creation:**
    *   Create custom challenges, categories, hints, and flags.
    *   Dynamic scoring challenges.
    *   Unlockable challenge support.
    *   Challenge plugin architecture for custom challenges.
    *   Static and Regex-based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads to the server or an Amazon S3-compatible backend.
    *   Limit challenge attempts & hide challenges.
    *   Automatic bruteforce protection
*   **Flexible Competition Modes:**
    *   Individual and team-based competitions.
    *   Allow users to compete solo or collaborate in teams.
*   **Comprehensive Scoreboard:**
    *   Automatic tie resolution.
    *   Ability to hide scores from the public.
    *   Freeze scores at a specific time.
*   **Engaging Visualization Tools:**
    *   Scoregraphs comparing the top 10 teams.
    *   Team progress graphs.
*   **Rich Content Management:**
    *   Markdown support for content creation.
*   **Communication and Notification:**
    *   SMTP and Mailgun email support.
    *   Email confirmation support.
    *   Forgot password support.
*   **Competition Management:**
    *   Automatic competition start and end times.
    *   Team management, hiding, and banning features.
*   **Customization:**
    *   Customize everything using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Importing and Exporting of CTF data for archiving.

## Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify `CTFd/config.ini` to fit your needs.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal to enable debug mode.

**Docker:**

You can quickly get started using Docker:

`docker run -p 8000:8000 -it ctfd/ctfd`

Or use Docker Compose:

`docker compose up`

For detailed deployment instructions and configuration options, refer to the [CTFd documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore the capabilities of CTFd by visiting the live demo:

*   [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get assistance and connect with the CTFd community:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or specialized projects, feel free to [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF deployment with managed hosting solutions available on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/), a powerful CTF statistics and event management platform. MLC offers event scheduling, team tracking, and single sign-on capabilities. Register your CTF event with MajorLeagueCyber to leverage these features, allowing users to automatically log in, track scores, submit writeups, and receive important event notifications.

Integrate with MajorLeagueCyber by adding your Client ID and Client Secret to `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
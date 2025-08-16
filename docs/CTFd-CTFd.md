# CTFd: The Open-Source Capture The Flag Framework

**CTFd is the ultimate open-source platform for hosting and running your own Capture The Flag (CTF) competitions, perfect for cybersecurity training and skill development.**  [Explore the original repository](https://github.com/CTFd/CTFd)

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features of CTFd:

*   **Customizable Challenges:**
    *   Create challenges with dynamic scoring.
    *   Implement unlockable challenge support.
    *   Leverage a plugin architecture for custom challenge types.
    *   Utilize static & regex-based flags, and create custom flag plugins.
    *   Offer unlockable hints to guide players.
    *   Allow file uploads to your server or S3-compatible backends.
    *   Control challenge attempts and visibility (hide/show).
*   **Competition Management:**
    *   Support for both individual and team-based competitions.
    *   Automated scoreboard with tie resolution.
    *   Option to hide scores from the public or freeze them at a specific time.
    *   Scoregraphs for the top teams and team progress visualization.
    *   Markdown content management for rich challenge descriptions.
*   **User & Communication Features:**
    *   SMTP and Mailgun email support for notifications and password resets.
    *   Automated competition start and end times.
    *   Team management options including hiding and banning.
*   **Extensibility & Customization:**
    *   Fully customizable with [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
    *   Import and export CTF data for archival and backups.
*   **Other notable features:**
    *   Automatic bruteforce protection
    *   And much more...

## Getting Started (Installation)

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Optionally, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify the `CTFd/config.ini` file to match your preferences.
3.  **Run:** Start the server using `python serve.py` or `flask run` in debug mode.

**Deployment Options:**

*   **Docker:** Use the pre-built Docker image:  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:**  Run `docker compose up` from the source repository.
*   Refer to the [CTFd docs](https://docs.ctfd.io/) for comprehensive [deployment options](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Check out a live demo of CTFd: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support & Community

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for help and discussions.
*   **Commercial Support:** For professional support or custom project assistance, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

Need a hassle-free CTFd experience? Explore managed hosting solutions on the [CTFd website](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.  Register your CTF event with MLC for features like automatic login, score tracking, writeup submissions, and event notifications.  Integrate by adding your client ID and secret in the `CTFd/config.py` or admin panel.

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo: [Laura Barbera](http://www.laurabb.com/)
*   Theme: [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound: [Terrence Martin](https://soundcloud.com/tj-martin-composer)
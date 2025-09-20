# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/mysql-ci.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is the premier open-source platform for hosting and running Capture The Flag (CTF) competitions, offering flexibility, ease of use, and extensive customization options.**

**[View the original repository on GitHub](https://github.com/CTFd/CTFd)**

## Key Features of CTFd

CTFd provides a comprehensive set of features to create, manage, and run engaging CTF events.

*   **Challenge Creation & Management:**
    *   Create custom challenges, categories, hints, and flags through an intuitive admin interface.
    *   Support for dynamic scoring challenges to keep the competition exciting.
    *   Unlockable challenge support for progressive difficulty.
    *   Plugin architecture for creating your own custom challenge types.
    *   Static & Regex based flags for diverse challenge implementations.
    *   Custom flag plugins for extended functionality.
    *   Unlockable hints to guide participants.
    *   File uploads to the server or an Amazon S3-compatible backend.
    *   Limit challenge attempts and hide challenges for enhanced security and control.
*   **Competition Modes:**
    *   Individual and team-based competitions to cater to various skill levels and preferences.
*   **Scoring & Leaderboard:**
    *   Automated scoreboard with tie resolution.
    *   Option to hide scores from the public for added suspense.
    *   Ability to freeze scores at a specific time to prevent late-game changes.
    *   Scoregraphs to visualize top team and individual progress.
*   **Content & Communication:**
    *   Markdown content management system for creating engaging content.
    *   SMTP + Mailgun email support for communication.
        *   Email confirmation and forgot password support.
*   **Competition Management:**
    *   Automatic competition starting and ending.
    *   Team management, including hiding and banning.
*   **Customization & Extensibility:**
    *   Highly customizable using [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Management:**
    *   Importing and exporting of CTF data for archival purposes.
*   **And More:** Features like bruteforce protection.

## Getting Started with CTFd

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configuration:** Modify `CTFd/config.ini` to configure your CTF instance.
3.  **Run the Application:**  Use `python serve.py` or `flask run` in a terminal to start the CTF in debug mode.

### Docker Support

Use the auto-generated Docker images with:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or, use Docker Compose:

```bash
docker compose up
```

### Resources

*   **Documentation:** [CTFd Documentation](https://docs.ctfd.io/)
*   **Deployment:** [Deployment Options](https://docs.ctfd.io/docs/deployment/installation)
*   **Getting Started:** [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)

## Live Demo

Experience CTFd in action: [CTFd Live Demo](https://demo.ctfd.io/)

## Support and Community

*   **Community Forum:** [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)
*   **Commercial Support:** [Contact Us](https://ctfd.io/contact/) for commercial support and custom project inquiries.

## Managed Hosting

For a hassle-free experience, consider [CTFd managed hosting](https://ctfd.io/) for fully managed deployments.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker. MLC provides event scheduling, team tracking, and single sign-on capabilities.  To integrate, register an account with MLC and input your client ID and client secret in the `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
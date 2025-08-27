# CTFd: The Open-Source Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/mysql.yml)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions/workflows/linting.yml)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and customizable open-source framework designed to help you easily create and manage your own Capture The Flag (CTF) competitions.**

CTFd provides a comprehensive platform for running CTFs, enabling you to engage participants and test their cybersecurity skills.  Check out the [CTFd repository](https://github.com/CTFd/CTFd) for the source code.

## Key Features of CTFd

*   **Easy Challenge Creation:**
    *   Create challenges, categories, hints, and flags through the admin interface.
    *   Supports dynamic scoring challenges for a more engaging experience.
    *   Implement unlockable challenges for a progressive CTF experience.
    *   Utilize a challenge plugin architecture to build custom challenges.
    *   Offers support for static & regex-based flags, along with custom flag plugins.
    *   Integrate unlockable hints to guide participants.
    *   Allows file uploads to the server or an Amazon S3-compatible backend.
    *   Set limits on challenge attempts and hide challenges as needed.
    *   Includes automatic brute-force protection.
*   **Flexible Competition Modes:**
    *   Support for individual and team-based competitions.
    *   Allows users to compete solo or form teams.
*   **Robust Scoring and Ranking:**
    *   Features a scoreboard with automatic tie resolution.
    *   Option to hide scores from the public for added suspense.
    *   Allows freezing scores at a specific time.
    *   Provides scoregraphs comparing the top 10 teams and team progress graphs.
*   **Content Management & Communication:**
    *   Uses a Markdown content management system for easy content creation.
    *   Offers SMTP and Mailgun email support for notifications and account management.
    *   Supports email confirmation and password reset functionalities.
*   **Competition Management:**
    *   Automate competition starting and ending times.
    *   Manage teams, including hiding and banning capabilities.
*   **Customization & Extensibility:**
    *   Customize everything using the plugin and theme interfaces for a personalized CTF experience.
*   **Data Management:**
    *   Import and export CTF data for archiving.
*   **And Much More!**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to your liking.
3.  **Run:** Use `python serve.py` or `flask run` in a terminal.

**Docker Deployment:**

You can quickly deploy CTFd using Docker:

```bash
docker run -p 8000:8000 -it ctfd/ctfd
```

Or use Docker Compose:

```bash
docker compose up
```

For detailed deployment instructions and a getting-started guide, see the [CTFd documentation](https://docs.ctfd.io/).

## Live Demo

Explore CTFd in action: [Live Demo](https://demo.ctfd.io/)

## Support

*   **Community:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for general support.
*   **Commercial Support:** Contact us for commercial support or special projects: [Contact](https://ctfd.io/contact/)

## Managed Hosting

For hassle-free CTF hosting, explore [CTFd's managed hosting solutions](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on. By registering your CTF event, users can automatically log in, track scores, submit writeups, and receive notifications.

To integrate with MajorLeagueCyber, register an account, create an event, and add the client ID and secret in the relevant section of `CTFd/config.py` or in the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
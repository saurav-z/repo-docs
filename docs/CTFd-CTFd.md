# CTFd: The Ultimate Capture The Flag (CTF) Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful and user-friendly framework designed to help you easily create and manage your own Capture The Flag (CTF) competitions.**  ([View the original repo](https://github.com/CTFd/CTFd))

## Key Features of CTFd:

CTFd offers a comprehensive suite of features to create engaging and customizable CTF experiences:

*   **Challenge Creation & Management:**
    *   Intuitive admin interface for creating challenges, categories, hints, and flags.
    *   Supports dynamic scoring challenges for added complexity.
    *   Offers unlockable challenge support to pace the CTF.
    *   Challenge plugin architecture for custom challenge development.
    *   Static and Regex-based flag options.
    *   Custom flag plugins for advanced flag types.
    *   Supports unlockable hints to guide participants.
    *   File uploads to the server or integrate with Amazon S3.
    *   Limit challenge attempts and hide challenges.
    *   Automatic brute-force protection to secure the CTF.
*   **Competition Structure:**
    *   Supports both individual and team-based competitions.
    *   Team management features, including hiding and banning.
*   **Scoring and Leaderboards:**
    *   Automatic tie resolution for fair scoring.
    *   Option to hide scores from the public.
    *   Ability to freeze scores at a specific time.
    *   Interactive scoregraphs comparing team progress.
*   **Content and Communication:**
    *   Built-in Markdown content management system for announcements and writeups.
    *   SMTP and Mailgun email support for notifications.
    *   Email confirmation and password reset functionalities.
    *   Automatic competition start and end times.
*   **Customization and Extensibility:**
    *   Extensive customization options through plugin and theme interfaces.
    *   Importing and exporting CTF data for archiving and portability.
*   **And many more features**

## Getting Started with CTFd:

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies.
2.  **Configure:** Modify `CTFd/config.ini` to customize your CTF settings.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

**Deployment Options:**
*   **Docker:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:** `docker compose up`

Refer to the [CTFd documentation](https://docs.ctfd.io/) for detailed [deployment instructions](https://docs.ctfd.io/docs/deployment/installation) and a [getting started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Experience CTFd firsthand:  [CTFd Demo](https://demo.ctfd.io/)

## Support and Community

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for community support.
*   For commercial support and custom projects, please [contact us](https://ctfd.io/contact/).

## Managed Hosting

For simplified deployments without managing infrastructure, explore [CTFd Managed Hosting](https://ctfd.io/).

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/) (MLC), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

*   Register your CTF event with MLC for automatic login and scoring features.
*   Integrate MLC by adding the client ID and client secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
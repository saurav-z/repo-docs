# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is the ultimate open-source platform to easily create, manage, and host your own Capture The Flag (CTF) competitions.**  

[View the original repository](https://github.com/CTFd/CTFd)

## Key Features of CTFd

CTFd provides a robust and customizable framework for running engaging CTF events. Here's what makes it stand out:

*   **Challenge Management:**
    *   Create and customize challenges, categories, hints, and flags through an intuitive admin interface.
    *   Supports dynamic scoring challenges for added complexity.
    *   Implement unlockable challenge features.
    *   Leverage a flexible challenge plugin architecture for custom challenges.
    *   Offers static and regex-based flags.
    *   Integrates custom flag plugins.
    *   Provides unlockable hints to assist players.
    *   Supports file uploads to server or Amazon S3-compatible backends.
    *   Includes options to limit challenge attempts and hide challenges.
    *   Offers automatic brute-force protection to secure challenges.
*   **Competition Modes:**
    *   Supports both individual and team-based competitions.
    *   Allows users to compete solo or form teams for collaborative play.
*   **Scoreboard & Rankings:**
    *   Features a scoreboard with automatic tie resolution.
    *   Offers the option to hide scores from the public.
    *   Allows freezing scores at a specific time for suspense.
    *   Includes score graphs to compare the top 10 teams and track team progress.
*   **Content & Communication:**
    *   Integrates a Markdown content management system for creating rich content.
    *   Provides SMTP and Mailgun email support for notifications and password resets.
    *   Enables email confirmation for user accounts.
    *   Supports automated competition starting and ending times.
*   **User & Team Management:**
    *   Offers comprehensive team management features, including hiding and banning capabilities.
*   **Customization:**
    *   Highly customizable through plugin and theme interfaces. [Plugins](https://docs.ctfd.io/docs/plugins/overview) and [Themes](https://docs.ctfd.io/docs/themes/overview)
*   **Data Management:**
    *   Allows importing and exporting CTF data for archiving purposes.
*   **And much more:** Explore a wide array of other features designed to create compelling CTF experiences.

## Getting Started: Installation

1.  **Install Dependencies:** Use `pip install -r requirements.txt` to install all necessary dependencies.
    *   Optionally, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configuration:** Modify the `CTFd/config.ini` file to align with your specific CTF requirements.
3.  **Run the Application:** Use `python serve.py` or `flask run` in a terminal to launch the application in debug mode.

**Docker Deployment:**

*   Use the pre-built Docker images with the following command: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Or use Docker Compose:  `docker compose up` (from the source repository).

For detailed deployment instructions and helpful guides, refer to the [CTFd Documentation](https://docs.ctfd.io/) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Experience CTFd in action at: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

For general support and community discussions, join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/): [![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)

For commercial support, or for any special project requirements, feel free to [contact us](https://ctfd.io/contact/).

## Managed Hosting

For those seeking to use CTFd without managing the underlying infrastructure, consider using a managed CTFd deployment.  Visit [the CTFd website](https://ctfd.io/) for more information.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/). MLC is a CTF stats tracker that offers features like event scheduling, team tracking, and single sign-on functionality.

By registering your CTF event with MajorLeagueCyber, participants can automatically log in, track individual and team scores, submit writeups, and receive important event notifications.

To integrate with MajorLeagueCyber, register for an account, create an event, and configure the client ID and client secret in the `CTFd/config.py` file or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
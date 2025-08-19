# CTFd: The Open-Source Capture The Flag Framework

**Ready to host your own cybersecurity competition?** CTFd is a user-friendly and highly customizable Capture The Flag (CTF) framework designed to get you up and running quickly.

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd

CTFd offers a comprehensive set of features to manage and run engaging CTF competitions:

*   **Challenge Management:**
    *   Create challenges, categories, hints, and flags through an intuitive admin interface.
    *   Support for dynamic scoring challenges, unlockable challenges, and custom challenge plugins.
    *   Flexible flag options: static, regex-based, and custom flag plugins.
    *   Implement unlockable hints and limit challenge attempts to fine-tune the difficulty.
    *   Control access with challenge hiding and file upload features (server or S3-compatible).
*   **Competition Modes:**
    *   Individual and team-based competition support.
    *   Automatic tie resolution on the scoreboard.
    *   Option to hide scores publicly and freeze scores at specific times.
*   **Engagement & Presentation:**
    *   Interactive scoreboards with graphs comparing top teams and team progress.
    *   Utilize a Markdown content management system for rich content.
*   **Communication & Integration:**
    *   SMTP + Mailgun email support for communication.
    *   Automated competition start and end times.
    *   Team management features including hiding and banning.
    *   Integrate with MajorLeagueCyber for enhanced features like event scheduling and SSO.
*   **Customization & Extensibility:**
    *   Highly customizable through [plugins](https://docs.ctfd.io/docs/plugins/overview) and [themes](https://docs.ctfd.io/docs/themes/overview).
    *   Import and export CTF data for archiving and backups.

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies via apt.
2.  **Configure:** Modify `CTFd/config.ini` to personalize your CTF setup.
3.  **Run:** Use `python serve.py` or `flask run` to launch in debug mode.

**Docker:**

*   Run the pre-built Docker image: `docker run -p 8000:8000 -it ctfd/ctfd`
*   Use Docker Compose: `docker compose up` (from the source repository)

**Resources:**

*   Refer to the [CTFd Documentation](https://docs.ctfd.io/) for installation, deployment, and in-depth guides.
*   Begin with the [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Explore a live demo of CTFd at: https://demo.ctfd.io/

## Support

Get help and connect with the community:

*   Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).

For commercial support and project-specific assistance, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF setup with managed CTFd deployments. Visit [the CTFd website](https://ctfd.io/) for more information.

## MajorLeagueCyber Integration

CTFd integrates seamlessly with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and single sign-on.

**Integration Steps:**

1.  Register your CTF event on MajorLeagueCyber.
2.  Enter the provided client ID and client secret within the `CTFd/config.py` file or the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
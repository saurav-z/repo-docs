# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/actions)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is a powerful, easy-to-use, and highly customizable open-source Capture The Flag (CTF) platform for cybersecurity enthusiasts and educators.**

[Visit the original repository on GitHub](https://github.com/CTFd/CTFd)

## Key Features of CTFd:

*   **Challenge Creation & Management:**
    *   Create custom challenges with ease via the admin interface.
    *   Supports dynamic scoring challenges for added complexity.
    *   Implement unlockable challenges to guide participants.
    *   Integrate challenge plugins for tailored CTF experiences.
    *   Utilize static & regex-based flags with custom plugins.
    *   Incorporate unlockable hints to assist players.
    *   Manage file uploads to the server or an S3-compatible backend.
    *   Limit challenge attempts and hide challenges for better control.

*   **Competition Structure:**
    *   Offers both individual and team-based competition modes.
    *   Includes an automatic tie resolution system for fairness.
    *   Hide scores and/or freeze them at a specific time for suspense.
    *   Provides scoregraphs and team progress graphs for visual tracking.

*   **Content & Communication:**
    *   Utilizes a Markdown content management system for flexibility.
    *   Offers SMTP and Mailgun email support for notifications.
    *   Includes email confirmation and forgot password support.

*   **Admin & Customization:**
    *   Automated competition starting and ending functionality.
    *   Provides team management tools (hiding, banning, etc.).
    *   Extensive customization options through plugin and theme interfaces.
    *   Importing and exporting CTF data for archiving and reuse.

## Getting Started with CTFd:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure CTFd:** Modify the [CTFd/config.ini](https://github.com/CTFd/CTFd/blob/master/CTFd/config.ini) file to suit your specific CTF requirements.
3.  **Run the Application:** Start the CTF server using `python serve.py` or `flask run` in a terminal (debug mode).

### Docker Deployment:

*   **Run with Docker:**
    ```bash
    docker run -p 8000:8000 -it ctfd/ctfd
    ```
*   **Docker Compose:**
    ```bash
    docker compose up
    ```

Consult the [CTFd documentation](https://docs.ctfd.io/) for in-depth [deployment](https://docs.ctfd.io/docs/deployment/installation) instructions and the [Getting Started guide](https://docs.ctfd.io/tutorials/getting-started/).

## Live Demo

Experience CTFd firsthand with the [live demo](https://demo.ctfd.io/).

## Support and Community

*   For basic support, connect with the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   For commercial support or project-specific needs, [contact us](https://ctfd.io/contact/).

## Managed Hosting

Simplify your CTF management with [managed CTFd deployments](https://ctfd.io/)

## Integration with MajorLeagueCyber

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a CTF stats tracker offering event scheduling, team tracking, and SSO.

To integrate, create an account on MajorLeagueCyber, create an event, and configure your CTFd instance with the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
# CTFd: The Premier Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)
![CTFd is a CTF in a can.](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)

**CTFd is an open-source, highly customizable Capture The Flag (CTF) platform designed to help you create and manage engaging cybersecurity competitions.** Looking to host your own CTF? Then look no further!

## Key Features of CTFd

*   **User-Friendly Interface:** Easily create and manage challenges, categories, hints, and flags through an intuitive admin panel.
*   **Dynamic Scoring:** Implement dynamic scoring challenges to keep your CTF exciting.
*   **Challenge Customization:**
    *   Plugin architecture allows for custom challenge types.
    *   Support for static and regular expression-based flags.
    *   Custom flag plugins.
    *   Unlockable hints.
    *   File uploads to the server or an Amazon S3-compatible backend.
    *   Limit challenge attempts & hide challenges.
    *   Automatic brute-force protection.
*   **Team and Individual Competitions:** Supports both individual and team-based competitions.
*   **Comprehensive Scoreboard:**
    *   Automatic tie resolution.
    *   Option to hide scores publicly.
    *   Score freezing at a specific time.
*   **Advanced Visualizations:** Scoregraphs comparing top teams and progress graphs.
*   **Content Management:** Markdown content management system for easy creation of challenge descriptions, announcements, and more.
*   **Email Integration:** SMTP and Mailgun email support, including email confirmation and password reset functionality.
*   **Competition Management:** Automatic start and end times for your CTF.
*   **Team Management:** Includes team management, hiding, and banning features.
*   **Extensive Customization:**  Easily customize everything using the [plugin](https://docs.ctfd.io/docs/plugins/overview) and [theme](https://docs.ctfd.io/docs/themes/overview) interfaces.
*   **Data Import/Export:** Import and export CTF data for archival or sharing.
*   **And much more!**

## Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify `CTFd/config.ini` to suit your needs.
3.  **Run the Server:** Use `python serve.py` or `flask run` to start the server in debug mode.

**Docker:**

*   **Run with Docker:** `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Run with Docker Compose:** `docker compose up` (from the source repository)

For detailed deployment options, consult the [CTFd documentation](https://docs.ctfd.io/docs/deployment/installation) and the [Getting Started](https://docs.ctfd.io/tutorials/getting-started/) guide.

## Live Demo

Explore CTFd firsthand: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support and Community

*   **Community Support:** Get help and connect with other CTFd users on the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/).
*   **Commercial Support:** For commercial support and custom projects, contact us via [https://ctfd.io/contact/](https://ctfd.io/contact/).

## Managed Hosting

Simplify CTF hosting with managed CTFd deployments. Visit [the CTFd website](https://ctfd.io/) for more details.

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber (MLC)](https://majorleaguecyber.org/), a CTF stats tracker for event scheduling, team tracking, and single sign-on.

**Integration Steps:**

1.  Register an account and create an event on MajorLeagueCyber.
2.  Install the client ID and secret in `CTFd/config.py` or within the admin panel:

    ```python
    OAUTH_CLIENT_ID = None
    OAUTH_CLIENT_SECRET = None
    ```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)

## Contribute

Want to help build a better CTFd? Visit the [CTFd GitHub repository](https://github.com/CTFd/CTFd).
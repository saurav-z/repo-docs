# CTFd: The Open-Source Capture The Flag Framework

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI)
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)](https://github.com/CTFd/CTFd/workflows/Linting)
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

**CTFd is the leading open-source platform to host and manage your own Capture The Flag (CTF) competitions, making it easy for cybersecurity enthusiasts to test their skills.**

[View the original repository on GitHub](https://github.com/CTFd/CTFd)

## What is CTFd?

CTFd is a flexible and user-friendly framework designed for running Capture The Flag competitions. Built with ease of use and customizability in mind, CTFd provides everything you need to create and manage engaging CTF events.  It's the perfect solution for cybersecurity training, education, and friendly competition.

![CTFd is a CTF in a can.](https://github.com/CTFd/CTFd/blob/master/CTFd/themes/core/static/img/scoreboard.png?raw=true)

## Key Features of CTFd:

*   **Intuitive Challenge Creation:** Easily create custom challenges, categories, hints, and flags through the admin interface.
*   **Dynamic Scoring:** Supports dynamic scoring challenges to keep the competition exciting.
*   **Challenge Variety:**  Offers a wide range of challenge types, including static, regex-based flags, and custom challenge plugins.
*   **Hint System:** Implement unlockable hints to guide participants.
*   **File Upload Support:**  Allows file uploads to the server or an Amazon S3-compatible backend.
*   **Anti-Bruteforce Measures:** Includes automatic bruteforce protection to secure your CTF.
*   **Team and Individual Competitions:**  Supports both individual and team-based competitions.
*   **Interactive Scoreboard:** Features an automatic tie resolution system and the option to hide or freeze scores.
*   **Visualizations:** Includes scoregraphs comparing top teams and progress graphs.
*   **Markdown Content Management:** Provides a built-in markdown content management system for rich content.
*   **Email Integration:** Offers SMTP and Mailgun email support for communication with participants.
*   **Competition Management:** Features automatic starting and ending times.
*   **Customization:** Highly customizable through plugin and theme interfaces.
*   **Data Management:** Supports importing and exporting CTF data for easy archiving and sharing.

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
    *   Alternatively, use the `prepare.sh` script to install system dependencies using `apt`.
2.  **Configure:** Modify [CTFd/config.ini](https://github.com/CTFd/CTFd/blob/master/CTFd/config.ini) to suit your needs.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

### Docker

Utilize pre-built Docker images for easy deployment:

`docker run -p 8000:8000 -it ctfd/ctfd`

Or use Docker Compose:

`docker compose up` (from the source repository)

### Resources

*   [CTFd Documentation](https://docs.ctfd.io/)
*   [Deployment Options](https://docs.ctfd.io/docs/deployment/installation)
*   [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/)

## Live Demo

Experience CTFd in action: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

Get help and connect with the community:

*   [MajorLeagueCyber Community](https://community.majorleaguecyber.org/)

For commercial support or project-specific assistance, [contact us](https://ctfd.io/contact/).

## Managed Hosting

For a hassle-free CTFd experience, explore managed deployments: [CTFd Website](https://ctfd.io/)

## MajorLeagueCyber Integration

CTFd seamlessly integrates with [MajorLeagueCyber](https://majorleaguecyber.org/), a platform for CTF event scheduling, team tracking, and single sign-on. Register your CTF event with MajorLeagueCyber to enable automatic user logins, score tracking, writeup submissions, and event notifications.

To integrate, register an account, create an event, and configure the client ID and secret in `CTFd/config.py` or the admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   Logo by [Laura Barbera](http://www.laurabb.com/)
*   Theme by [Christopher Thompson](https://github.com/breadchris)
*   Notification Sound by [Terrence Martin](https://soundcloud.com/tj-martin-composer)
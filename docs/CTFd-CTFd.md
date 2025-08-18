# CTFd: The Open-Source Capture The Flag Framework for Cyber Security Training

CTFd is a powerful and flexible open-source platform designed to host and manage Capture The Flag (CTF) competitions, perfect for cybersecurity enthusiasts, educators, and organizations. (**[View the original repository](https://github.com/CTFd/CTFd)**)

[![CTFd MySQL CI](https://github.com/CTFd/CTFd/workflows/CTFd%20MySQL%20CI/badge.svg?branch=master)]
[![Linting](https://github.com/CTFd/CTFd/workflows/Linting/badge.svg?branch=master)]
[![MajorLeagueCyber Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fcommunity.majorleaguecyber.org%2F)](https://community.majorleaguecyber.org/)
[![Documentation Status](https://api.netlify.com/api/v1/badges/6d10883a-77bb-45c1-a003-22ce1284190e/deploy-status)](https://docs.ctfd.io)

## Key Features

CTFd provides a comprehensive set of features to create engaging and effective CTF experiences:

*   **Intuitive Admin Interface:** Easily create and manage challenges, categories, hints, and flags.
*   **Dynamic Scoring:** Implement dynamic scoring challenges to keep participants engaged.
*   **Challenge Customization:** Utilize a plugin architecture to create custom challenges tailored to your needs.
*   **Flexible Flag Support:** Supports static and regex-based flags, along with custom flag plugins.
*   **Hint System:** Offer unlockable hints to guide participants.
*   **File Uploads:** Allow file uploads to the server or integrate with S3-compatible backends.
*   **Challenge Controls:** Limit challenge attempts and hide challenges.
*   **Team and Individual Competitions:** Support both individual and team-based participation.
*   **Real-time Scoreboard:** Provides a live scoreboard with automatic tie resolution.
*   **Scoreboard Options:** Hide or freeze scores at specific times for competitive integrity.
*   **Visualizations:** Display scoregraphs and team progress graphs.
*   **Markdown Content Management:** Easily create rich content using Markdown.
*   **Email Integration:** Supports SMTP and Mailgun for email notifications and password resets.
*   **Competition Management:** Automate competition start and end times.
*   **User and Team Management:** Manage teams, hide teams, and ban users.
*   **Customization:** Fully customize your CTF environment through plugins and themes.
*   **Data Import/Export:** Import and export CTF data for archival and sharing.
*   **And much more!**

## Getting Started

### Installation

1.  **Install dependencies:** `pip install -r requirements.txt`
    *   You can also use the `prepare.sh` script to install system dependencies using apt.
2.  **Configure:** Modify the `CTFd/config.ini` file to your liking.
3.  **Run:** Use `python serve.py` or `flask run` to start in debug mode.

### Deployment Options

*   **Docker:** Use the pre-built Docker image:  `docker run -p 8000:8000 -it ctfd/ctfd`
*   **Docker Compose:** Use `docker compose up` from the source repository.
*   **Detailed Installation:** Consult the [CTFd Documentation](https://docs.ctfd.io/docs/deployment/installation) for comprehensive deployment instructions.
*   **Getting Started Guide:** Refer to the [Getting Started Guide](https://docs.ctfd.io/tutorials/getting-started/) for a quick setup.

## Live Demo

Explore the live demo to experience CTFd in action: [https://demo.ctfd.io/](https://demo.ctfd.io/)

## Support

*   **Community Support:** Join the [MajorLeagueCyber Community](https://community.majorleaguecyber.org/) for basic support.
*   **Commercial Support:** Contact us for commercial support or special projects via the [CTFd website](https://ctfd.io/contact/).

## Managed Hosting

For hassle-free CTFd deployments, consider managed hosting solutions: Check out [the CTFd website](https://ctfd.io/) for managed CTFd deployments.

## MajorLeagueCyber Integration

CTFd is deeply integrated with [MajorLeagueCyber](https://majorleaguecyber.org/) (MLC), a CTF statistics tracker.  MLC provides:

*   Event scheduling
*   Team tracking
*   Single sign-on

**Integration Steps:**

1.  Register for an account on MajorLeagueCyber.
2.  Create a CTF event.
3.  Install the client ID and secret in your `CTFd/config.py` or admin panel:

```python
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
```

## Credits

*   **Logo:** [Laura Barbera](http://www.laurabb.com/)
*   **Theme:** [Christopher Thompson](https://github.com/breadchris)
*   **Notification Sound:** [Terrence Martin](https://soundcloud.com/tj-martin-composer)
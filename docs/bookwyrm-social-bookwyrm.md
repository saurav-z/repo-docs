# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, connect with other bookworms, and discuss your favorite reads.** Check out the original repo [here](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books with your network.
*   **Reading Tracker:** Keep a record of the books you've read, are reading, and want to read.
*   **ActivityPub Federation:** Interact with users on other BookWyrm instances and other ActivityPub services like Mastodon and Pleroma, creating a decentralized reading community.
*   **Community-Driven:** Join or create book clubs and communities based on your interests.
*   **Privacy Controls:** Manage your posts' visibility and choose which instances to federate with.
*   **Decentralized:** Take control of your social reading experience with a platform free from corporate control.

## Federation and Community

BookWyrm utilizes ActivityPub to connect users across different instances and services. This federation allows for small, self-governing communities, in contrast to centralized platforms.  Create a community around your interests or join existing ones, choosing your own moderation and instance rules.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support BookWyrm](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Technology Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker-compose, Gunicorn, Flower, Nginx

## Set Up BookWyrm

Detailed instructions for setting up BookWyrm in both development and production environments can be found in the [documentation](https://docs.joinbookwyrm.com/).

## Contributing

Want to help improve BookWyrm?  Find out how to get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
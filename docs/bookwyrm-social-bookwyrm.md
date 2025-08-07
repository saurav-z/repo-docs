# BookWyrm: The Social Network for Book Lovers

BookWyrm is a social network for book lovers, allowing you to track your reading, connect with friends, and discover new books, all while prioritizing privacy and community. [View the original repository on GitHub](https://github.com/bookwyrm-social/bookwyrm).

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and discuss books with other readers.
*   **Reading Tracking:** Keep a record of books you've read, are currently reading, and want to read.
*   **ActivityPub Federation:** Connect with users on other BookWyrm instances and compatible platforms like Mastodon and Pleroma, fostering a decentralized social experience.
*   **Privacy and Moderation:** Control your posts' visibility and manage your network for a safe and personalized reading experience.

## About BookWyrm

BookWyrm provides a platform for social reading, enabling you to track your reading, review books, and connect with fellow book enthusiasts. Built on ActivityPub, it embraces federation to create small, self-governed communities that prioritize user control and privacy, offering a refreshing alternative to centralized social networks.

## Federation

BookWyrm utilizes ActivityPub for federation, allowing interoperability with other BookWyrm instances and ActivityPub-compliant services like Mastodon. This enables you to join communities tailored to your interests and interact with users across the fediverse. Learn more about BookWyrm's implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Set up BookWyrm

Visit the [documentation website](https://docs.joinbookwyrm.com/) for instructions on setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Join the BookWyrm community and contribute to its growth! Find out how to get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
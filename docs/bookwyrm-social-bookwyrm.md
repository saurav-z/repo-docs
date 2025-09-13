# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network that empowers you to connect with other readers, share your thoughts, and discover new books in a privacy-focused and community-driven environment.** [Visit the original repository](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub release (latest SemVer)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share & Discuss Books:** Write and share book reviews, post quotes, and engage in conversations with other readers across the network.
*   **Track Your Reading:** Easily keep track of what you're currently reading, have read, and want to read in the future.
*   **Federation with ActivityPub:** Connect with users on other BookWyrm instances and services like Mastodon and Pleroma, fostering a decentralized social reading experience.  Share and collaboratively build a database of books.
*   **Privacy & Moderation:** Control your privacy settings and choose which instances to federate with, allowing for a personalized and safe online experience.
*   **Community-Driven:** Join or create small, trusted communities focused on specific interests, offering a refreshing alternative to centralized platforms.

## About BookWyrm

BookWyrm is a social platform designed for book lovers to connect, share, and discover new literature. It emphasizes community-building and provides a space for in-depth discussions about books. BookWyrm is built on ActivityPub, enabling it to interoperate with other ActivityPub-compliant services, allowing for greater interoperability and choice.

## Federation

BookWyrm is built on the ActivityPub protocol. ActivityPub allows you to interact with users on different BookWyrm instances and other ActivityPub-compliant services like Mastodon. Federation creates a network of independent instances where users can form small, self-determining communities, as opposed to monolithic services.

Developers can learn more about BookWyrm's implementation via [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker and docker-compose, Gunicorn, Flower, Nginx

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)

## Set Up BookWyrm

Find detailed instructions on how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) on the documentation website.

## Contributing

Contribute to BookWyrm and help grow the project! Learn how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)
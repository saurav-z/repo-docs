# BookWyrm: A Social Network for Readers (Decentralized & Federated)

**Tired of corporate social media silos? BookWyrm is a decentralized social network built for readers, offering a privacy-focused and federated experience for book lovers.**  [View the source on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm is a powerful, open-source platform for readers who want to connect, share their thoughts on books, and discover new reads within a community-driven environment.  It utilizes the ActivityPub protocol, enabling seamless federation with other BookWyrm instances and services like Mastodon and Pleroma, fostering a truly decentralized and interconnected reading experience.

## Key Features

*   **Share & Discuss Books:** Write reviews, post quotes, and engage in conversations with other readers about your favorite books.
*   **Track Your Reading:** Easily keep track of books you've read, are currently reading, and want to read in the future.
*   **Federated Network:** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma. Share and discover books across a decentralized network.
*   **Privacy & Moderation:** Enjoy control over your data and interactions with robust privacy settings and community moderation tools.
*   **Discover New Reads:** Explore books and authors recommended by your network, collaboratively building a decentralized book database.

## Federation Explained

BookWyrm leverages the power of [ActivityPub](http://activitypub.rocks/) to create a federated social network. This means:

*   **Interoperability:** BookWyrm instances can connect with each other, and with other ActivityPub services like Mastodon and Pleroma.
*   **Community Control:**  Run your own BookWyrm instance for a book club, a group of friends, or any community you choose.  Each instance maintains its autonomy and can choose who to connect with, fostering a high-trust environment.
*   **Decentralized Database:** Shared metadata about books and authors contributes to a decentralized book database, expanding your reading discovery.

## Tech Stack

**Backend:**

*   Django (Web Server)
*   PostgreSQL (Database)
*   ActivityPub (Federation Protocol)
*   Celery (Task Queuing)
*   Redis (Task Backend & Activity Stream Manager)

**Frontend:**

*   Django Templates
*   Bulma.io (CSS Framework)
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker-Compose
*   Gunicorn (Web Runner)
*   Flower (Celery Monitoring)
*   Nginx (HTTP Server)

## Get Started

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:**  [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon Profile:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

The documentation provides instructions on setting up BookWyrm in a developer environment or for production use.

## Contribute

Contribute to the BookWyrm project!  See [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) for details on how you can get involved.
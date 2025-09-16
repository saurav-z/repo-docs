# BookWyrm: A Social Network for Book Lovers (Decentralized Reading)

BookWyrm is a free and open-source social network that empowers you to connect with fellow readers, share your thoughts on books, and discover new titles in a decentralized and federated environment.  Learn more at the [original repo](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)


## Key Features

*   **Share Your Reading Journey:** Post reviews, comments, and quotes from your favorite books, and discuss them with others.
*   **Track Your Reading:**  Keep a record of the books you've read, are currently reading, and want to read.
*   **Decentralized & Federated:**  Connect with users on other BookWyrm instances and other ActivityPub-compatible platforms like Mastodon and Pleroma, fostering a community-driven ecosystem.
*   **Control Your Privacy & Moderation:**  Manage your visibility settings and choose which instances you want to connect with, giving you greater control over your online experience.
*   **Discover New Books:** Explore a growing, collaborative database of books and authors.

## How BookWyrm Works: Federation Explained

BookWyrm utilizes ActivityPub, a decentralized social networking protocol.  This means:

*   **Interoperability:** BookWyrm instances can seamlessly communicate with each other.
*   **Cross-Platform Compatibility:** You can interact with users on other ActivityPub platforms, like Mastodon.
*   **Community Control:**  Create or join smaller, self-governed communities, offering a contrast to monolithic platforms.
*   **Open Collaboration:**  Federation allows for the collective building of a decentralized book and author database.

## Useful Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Tech Stack

BookWyrm is built using a robust and modern tech stack:

**Backend:**

*   Django (Web Server)
*   PostgreSQL (Database)
*   ActivityPub (Federation)
*   Celery (Task Queuing)
*   Redis (Task Backend & Activity Stream Manager)

**Frontend:**

*   Django Templates
*   Bulma.io (CSS Framework)
*   Vanilla JavaScript

**Deployment:**

*   Docker & Docker Compose
*   Gunicorn (Web Runner)
*   Flower (Celery Monitoring)
*   Nginx (HTTP Server)

## Get Started with BookWyrm

Detailed instructions for setting up BookWyrm are available on the [documentation website](https://docs.joinbookwyrm.com/), covering both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production deployments](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Help improve BookWyrm!  Learn how to get involved and contribute to the project by visiting [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
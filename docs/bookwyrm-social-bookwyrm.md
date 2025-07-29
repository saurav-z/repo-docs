# BookWyrm: A Social Network for Book Lovers

BookWyrm is a decentralized social network where you can track your reading, share reviews, and connect with fellow bookworms in a federated and privacy-focused environment.  Dive deeper into the project at the [original repository](https://github.com/bookwyrm-social/bookwyrm).

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading & Reviews:** Share your thoughts on books through reviews, quotes, and discussions.
*   **Reading Activity Tracking:** Keep a record of the books you've read, are currently reading, and want to read.
*   **ActivityPub Federation:** Connect with users on other BookWyrm instances and compatible services like Mastodon and Pleroma, building a collaborative, decentralized book database.
*   **Privacy & Moderation:** Control your post visibility and choose which instances you interact with.
*   **Community Focused:** Join or create your own BookWyrm instance, fostering a sense of belonging and shared interest.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)

## About BookWyrm

BookWyrm is designed for social interaction around books, including features for tracking your reading progress, writing book reviews, and connecting with friends. It leverages ActivityPub for federation, enabling interoperability with other platforms and fostering a decentralized social experience. This allows for smaller, more intimate communities centered around shared interests, offering an alternative to centralized platforms.

## Federation Explained

BookWyrm uses ActivityPub to connect with other BookWyrm instances and compatible platforms, allowing you to:

*   Follow and interact with users on other ActivityPub services like Mastodon.
*   Build a shared, decentralized database of books and authors through collaborative metadata sharing.
*   Create or join smaller, self-governing communities based on shared interests.

Learn more about BookWyrm's ActivityPub implementation in `FEDERATION.md`.

## Tech Stack

**Web Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend & activity stream manager

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and docker-compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Get Started with BookWyrm

Comprehensive instructions for setting up BookWyrm are available in the documentation, including guides for both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production deployments](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Contribute to the BookWyrm project!  Find out how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
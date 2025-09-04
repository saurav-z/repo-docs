# BookWyrm: A Social Network for Book Lovers 

**BookWyrm is a decentralized social network that empowers book enthusiasts to connect, share reviews, and discover new reads.**  Discover the power of a social network built by bookworms, for bookworms.

[View the original repository on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share book reviews, discuss your current reads, and engage in conversations with fellow bookworms.
*   **Reading Tracking:** Keep a record of the books you've read, are currently reading, and want to read in the future.
*   **Federated Network:** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma, fostering a decentralized reading community.
*   **Privacy & Moderation:** Enjoy control over your posts' visibility and the instances you choose to connect with, promoting a safe and tailored social experience.
*   **Discover New Books:** Build a collaborative database of books and authors through federation.

## Why BookWyrm?

BookWyrm is more than just a social network; it's a platform for building small, trusted communities centered around the joy of reading. Unlike monolithic platforms, BookWyrm allows you to create or join instances focused on specific interests or groups, offering autonomy and control over your social experience.

## Federation Explained

BookWyrm utilizes the ActivityPub protocol to connect with other platforms. This allows you to:

*   Interact with users on other BookWyrm instances.
*   Share and view reviews and discussions across different servers.
*   Connect with users on other ActivityPub services like Mastodon.

Developers can find more information on BookWyrm's implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

BookWyrm is built using a modern tech stack:

*   **Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis.
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript.
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx.

## Get Started with BookWyrm

Visit the [documentation website](https://docs.joinbookwyrm.com/) for detailed instructions on setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute to BookWyrm

We welcome contributions from everyone! Learn how you can contribute to the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Links

*   [Project homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)
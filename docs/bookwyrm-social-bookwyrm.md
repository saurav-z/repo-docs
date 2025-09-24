# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you connect with friends, track your reading, and discuss books, all while maintaining control over your data.** ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share book reviews, quotes, and reading updates with your network.
*   **Reading Activity Tracking:** Keep a record of books you've read, want to read, and are currently reading.
*   **Decentralized Federation (ActivityPub):** Connect with users on other BookWyrm instances and compatible platforms like Mastodon and Pleroma.
*   **Privacy Controls:** Manage your post visibility and choose which instances to federate with.
*   **Community Building:** Create or join book clubs and connect with like-minded readers.
*   **Decentralized Database:** Contribute to a collaborative, decentralized book database.

## Why Choose BookWyrm?

BookWyrm offers a refreshing alternative to centralized social networks, empowering you to:

*   **Control Your Data:** Own your reading data and choose where it resides.
*   **Join Independent Communities:** Connect with friends and groups in a self-governed environment.
*   **Discover New Books:** Explore recommendations and discussions from your network.

## Federation Explained

BookWyrm utilizes the ActivityPub protocol, enabling federation. This means:

*   You can interact with users on other BookWyrm instances.
*   You can connect with users on other ActivityPub-compatible platforms, such as Mastodon.
*   Your data is not locked into a single platform.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Technology Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io (CSS), Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Get Started

Refer to the [documentation](https://docs.joinbookwyrm.com/) for instructions on setting up BookWyrm, whether in a developer environment or for production use.

## Contribute

Help make BookWyrm even better! Learn how to contribute at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
# BookWyrm: The Social Network for Readers 📚

BookWyrm is a decentralized social network that lets you track your reading, connect with other book lovers, and discuss your favorite stories, all while maintaining control over your data. Check out the original repository [here](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, discuss books, post quotes, and engage in conversations with other readers.
*   **Reading Tracker:** Keep a record of books you've read, are currently reading, and want to read.
*   **ActivityPub Federation:** Interact with users on other BookWyrm instances and compatible services like Mastodon, fostering a decentralized network.
*   **Privacy and Moderation:** Control who sees your posts and manage your instance's federation with other communities.

## About BookWyrm

BookWyrm is a social platform designed for readers. It allows users to track their reading progress, review books, follow friends, and discover new titles. Built on ActivityPub, BookWyrm enables federation, connecting you with diverse communities and services across the web.

## Federation: Connect with the Fediverse

BookWyrm is built on ActivityPub, which enables:

*   **Interoperability:** Communicate with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma.
*   **Community-Driven:** Join small, trusted communities, focused on specific interests or groups, fostering a sense of autonomy.
*   **Decentralized Network:** Build a collaborative database of books and authors through shared metadata.

Developers can find more details about BookWyrm's federation implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Technology Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (for task queue and activity stream)
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Get Started with BookWyrm

Comprehensive setup instructions are available on the [documentation website](https://docs.joinbookwyrm.com/), including guides for [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production setups](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute to BookWyrm

There are many ways to help improve and support the BookWyrm project! Learn more about how you can get involved in [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Stay Connected

[![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
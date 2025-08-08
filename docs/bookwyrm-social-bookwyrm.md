# BookWyrm: The Social Network for Bookworms

**BookWyrm is a decentralized, open-source social network designed for readers to connect, share reviews, and track their reading journeys.** ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Connect & Discuss:** Share reviews, discuss books, and engage in conversations with other readers.
*   **Reading Tracking:** Keep a record of books you've read, are currently reading, and want to read.
*   **Federated Network:** Interact with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma.
*   **Decentralized & Community-Focused:** Join or create your own instance for a more focused, community-driven experience.
*   **Privacy & Moderation:** Control your privacy settings and choose which instances you federate with.
*   **Post about books:** Compose reviews, comment on what you're reading, and post quotes from books. You can converse with other BookWyrm users across the network about what they're reading.
*   **Track reading activity:** Keep track of what books you've read, and what books you'd like to read in the future.
*   **Federation with ActivityPub:** Federation allows you to interact with users on other instances and services, and also shares metadata about books and authors, which collaboratively builds a decentralized database of books.
*   **Privacy and moderation:** Users and administrators can control who can see their posts and what other instances to federate with.

## About BookWyrm

BookWyrm is built on the principles of social reading and community, offering a platform for readers to connect and share their passion for books. It's designed for social interaction and offers robust features for tracking your reading and interacting with others.

## Federation

BookWyrm uses ActivityPub, enabling it to interact with other instances of BookWyrm and other ActivityPub services, creating a decentralized network.  This allows for the creation of small, self-governing communities, in contrast to centralized platforms.

For developers interested in BookWyrm's ActivityPub implementation, see [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

BookWyrm is built using a modern tech stack, including:

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis.
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript.
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)
*   [Contribute](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)

## Getting Started

The [documentation website](https://docs.joinbookwyrm.com/) provides detailed instructions on setting up BookWyrm, whether in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or for [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute

Join the BookWyrm community and contribute to the project's growth! Find out how to get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
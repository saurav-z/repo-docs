# BookWyrm: The Social Network for Readers

**BookWyrm is a decentralized social network that empowers readers to connect, share, and discuss books in a privacy-focused and community-driven environment.** ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Write reviews, comment on books, share quotes, and engage in discussions with fellow book lovers.
*   **Reading Tracking:** Easily track the books you've read, are currently reading, and want to read in the future.
*   **Federated Network:** Connect with users on other BookWyrm instances and other ActivityPub services like Mastodon and Pleroma, fostering a decentralized reading community.
*   **Community Control:** Join or create small, self-governed communities that can tailor their experience, set their own moderation rules, and decide who to connect with.
*   **Privacy and Moderation:** Enjoy control over your posts and who sees them, along with administrator-level moderation tools.

## About BookWyrm

BookWyrm is a social reading platform designed to help you discover new books, share your thoughts, and connect with other readers. It provides a space for reading, reviewing, and finding your next favorite book, with a focus on user privacy and community-driven content. Unlike centralized platforms, BookWyrm prioritizes decentralized networking through ActivityPub, allowing users to connect with others across the fediverse.

## Federation: Connect with the Fediverse

BookWyrm is built on the ActivityPub protocol, enabling seamless interaction with other BookWyrm instances and platforms like Mastodon and Pleroma. This allows for small, independent communities and a collaborative, decentralized ecosystem, where users control their experience. Learn more about how BookWyrm implements ActivityPub in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Get Started

Visit the [BookWyrm documentation website](https://docs.joinbookwyrm.com/) to learn how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)

## Contribute

Contribute to BookWyrm and help build the next generation of social reading.  Learn how to contribute at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
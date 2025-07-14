# BookWyrm: A Social Network for Readers 📚

BookWyrm is a decentralized social network that lets you connect with other readers, share reviews, and track your reading journey.  Check out the original repository [here](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Post reviews, quotes, and comments about books you're reading.
*   **Reading Tracking:** Keep a record of books you've read and create a reading list for future reads.
*   **Federated Network:** Interact with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma.
*   **Decentralized & Community-Focused:** Build and join small, trusted communities with control over your privacy and moderation.
*   **Privacy Controls:** Manage who sees your posts and choose which instances to federate with.

## About BookWyrm

BookWyrm is a social reading platform built on the ActivityPub protocol, promoting a decentralized and community-driven approach to social networking. Unlike centralized platforms, BookWyrm emphasizes small, self-governing communities where users have more control over their experience.

## Federation with ActivityPub

BookWyrm utilizes ActivityPub, allowing seamless interaction between different BookWyrm instances and other ActivityPub-compliant services such as Mastodon. This means you can connect with friends regardless of their preferred platform, share reviews, and build a collaborative, decentralized book database.  Learn more about BookWyrm's federation implementation in `FEDERATION.md`.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io CSS Framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Getting Started

Detailed installation instructions for both development and production environments can be found on the [documentation website](https://docs.joinbookwyrm.com/).

## Contributing

BookWyrm welcomes contributions from everyone! Find out how you can join the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)
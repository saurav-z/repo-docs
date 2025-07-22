# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, share reviews, and connect with other book enthusiasts, fostering a vibrant community around literature.**  [Explore BookWyrm on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading & Reviews:** Share your thoughts, write reviews, and engage in discussions about books with other readers.
*   **Reading Activity Tracking:**  Keep a personal record of your reading journey, including books you've read, are currently reading, and want to read.
*   **Federation via ActivityPub:**  Interact with users on other BookWyrm instances and other ActivityPub-compatible platforms like Mastodon and Pleroma, creating a wider network.
*   **Decentralized Community:**  Join or create your own self-governed reading communities, fostering trust and focused conversations.
*   **Privacy & Moderation:** Control your visibility and curate your federated network to ensure a safe and enjoyable experience.

## About BookWyrm

BookWyrm is a platform designed for social reading, allowing you to track your reading habits, review books, and follow fellow bookworms. Unlike platforms primarily focused on cataloging, BookWyrm combines cataloging with social networking in a decentralized, community-driven environment.

## Federation Explained

BookWyrm utilizes [ActivityPub](http://activitypub.rocks/) for federation, enabling interoperability with other BookWyrm instances and compatible platforms such as Mastodon and Pleroma.  This architecture allows for small, independent, and trusted communities, promoting a sense of belonging and control compared to monolithic social networks.

## Tech Stack

**Web Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend & activity stream manager

**Front End:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and docker-compose
*   Gunicorn web runner
*   Flower Celery monitoring
*   Nginx HTTP server

## Get Started

Comprehensive setup instructions are available in the official [documentation](https://docs.joinbookwyrm.com/), including guidance for both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production deployments](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute

Help build and improve BookWyrm!  Learn how you can contribute to the project by reviewing the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file.
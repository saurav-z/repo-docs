# BookWyrm: A Social Network for Book Lovers (and Federation!)

BookWyrm is a decentralized social network that lets you track your reading, share reviews, and connect with fellow bookworms, all while embracing the power of federation. ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share & Discuss Books:** Write reviews, post quotes, and engage in conversations about your favorite books and authors.
*   **Track Your Reading:** Easily log what you're reading, what you've read, and your to-read list.
*   **Federated Social Network:** Interact with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma, fostering a decentralized community.
*   **Decentralized Database:** Collaborate with others to build a shared, distributed database of books and authors.
*   **Privacy & Control:** Manage your privacy settings and control which instances you federate with.

## Why Choose BookWyrm?

BookWyrm offers a refreshing alternative to centralized book-focused social networks, emphasizing community-driven content and user autonomy.  Build your own book club, join a specific interest group, or connect with friends – all within a network that prioritizes your control.

## Federation Explained

BookWyrm leverages ActivityPub, a decentralized social networking protocol, to connect with other instances. This federation allows you to:

*   Follow and interact with users on different BookWyrm instances and other ActivityPub platforms.
*   Share book reviews and discussions across the network.
*   Contribute to a collectively built database of books and authors.
*   Join and foster small, self-determining communities.

## Links & Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Tech Stack

**Backend:**

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

*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Get Started

Find detailed instructions on setting up BookWyrm in both a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) on the [documentation website](https://docs.joinbookwyrm.com/).

## Contribute

Join the BookWyrm community! Learn more about contributing to the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
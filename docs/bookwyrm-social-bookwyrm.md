# BookWyrm: Your Decentralized Social Network for Book Lovers

BookWyrm is a powerful social network designed to connect readers, share reviews, and discover new books in a federated, community-driven environment. [Check out the original repository here](https://github.com/bookwyrm-social/bookwyrm).

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Connect and Share:** Post reviews, discuss books, share quotes, and engage with other readers.
*   **Track Your Reading:** Effortlessly keep track of what you're reading, have read, and plan to read.
*   **Federated Network:** Interact with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma, fostering a decentralized community.
*   **Community-Driven:** Join or create your own instance, fostering small, trusted communities with control over moderation and federation.
*   **Privacy & Control:** Manage your posts' visibility and choose which instances to connect with.

## About BookWyrm

BookWyrm is a social platform built for readers, allowing you to track your reading, review books, and connect with fellow book enthusiasts. Built on ActivityPub, it fosters interoperability with other services, promoting a decentralized and community-focused environment, unlike centralized platforms.

## Federation Explained

BookWyrm uses ActivityPub to communicate with other instances and platforms, like Mastodon. This means you can connect with friends on other servers, share reviews across networks, and build a distributed database of books and authors. Run your own instance for a book club or connect with the broader network to discover new reads and perspectives.

## Tech Stack

**Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub for federation
*   Celery task queuing
*   Redis task backend
*   Redis activity stream manager

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and docker-compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)
*   [FEDERATION.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md)
*   [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)

## Get Started

Learn how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) on the documentation website.

## Contributing

Interested in contributing? Find out how to get involved with the BookWyrm project by checking out the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file.
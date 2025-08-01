# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network built for readers to connect, share reviews, and discover new books, fostering a vibrant community around literature.** [(Original Repository)](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, discuss books, and connect with other bookworms.
*   **Track Your Reading:** Keep a record of what you've read, what you're currently reading, and your reading wishlist.
*   **Federated Network (ActivityPub):** Interact with users on other BookWyrm instances and compatible platforms like Mastodon, fostering a decentralized network of readers.
*   **Privacy & Moderation:** Control your data and who can see your posts, with options for instance-level moderation.
*   **Discover New Books:** Explore a collaboratively built, decentralized database of books, authors, and reviews.

## About BookWyrm

BookWyrm is a social platform designed specifically for readers.  It focuses on fostering social interaction around books.

## Federation and ActivityPub

BookWyrm is built on the ActivityPub protocol, enabling interoperability with other BookWyrm instances and services like Mastodon and Pleroma. This allows users to form small, independent communities while remaining connected within a larger, decentralized network. Learn more about ActivityPub and the philosophy of decentralized social networking at [https://runyourown.social/](https://runyourown.social/).

Developers can find details about BookWyrm's ActivityPub implementation at [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

BookWyrm uses a robust and modern tech stack:

**Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend and activity stream manager

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and docker-compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)

## Get Started

Detailed instructions for setting up BookWyrm are available in the [documentation](https://docs.joinbookwyrm.com/), including guides for both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) setups.

## Contributing

The BookWyrm project welcomes contributions from everyone! Learn how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
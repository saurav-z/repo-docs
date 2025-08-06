# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network where you can track your reading, connect with friends, and discuss books in a privacy-focused, federated environment.**  ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers you to discover new books, share your thoughts, and build a community around your passion for reading. Built on ActivityPub, it connects you with other BookWyrm instances and platforms like Mastodon and Pleroma.

## Key Features

*   **Share & Discuss Books:** Write reviews, post quotes, comment on what you're reading, and engage in conversations about books with fellow BookWyrm users.
*   **Track Your Reading:**  Maintain a reading log, mark books you've finished, and create a "to-read" list.
*   **Federated Network:**  Interact with users on other ActivityPub-compatible platforms (like Mastodon), fostering a decentralized and interconnected book-loving community.
*   **Privacy & Moderation:**  Control your privacy settings and curate the instances you connect with. Build trusted communities.
*   **Decentralized Book Database:** Leverage a growing, collaborative database of books and authors shared across the network.

## About BookWyrm

BookWyrm is a social reading platform designed for readers.  You can use it to track your reading, review books, and follow your friends.

## Federation Explained

BookWyrm uses [ActivityPub](http://activitypub.rocks/) for federation. This technology allows BookWyrm instances to communicate with each other and other ActivityPub-compliant services like Mastodon. This interoperability means you can connect with friends across different platforms and create specialized communities focused on particular interests.

## Tech Stack

**Backend:**
*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend
*   Redis activity stream manager

**Frontend:**
*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**
*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower Celery monitoring
*   Nginx HTTP server

## Get Started

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:**  Detailed installation and usage instructions are available at [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/).

## Support

*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)

## Contribute

Interested in helping out?  Learn how you can contribute to the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
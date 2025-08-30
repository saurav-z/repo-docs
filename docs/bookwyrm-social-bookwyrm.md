# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, connect with fellow bookworms, and discover new literary adventures.** Learn more and contribute on the [original BookWyrm repository](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest SemVer)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share Book Reviews & Recommendations:** Write and share reviews, discuss books, and post favorite quotes.
*   **Track Your Reading Journey:** Easily log the books you've read, are currently reading, and want to read.
*   **Decentralized Social Reading:** Built on ActivityPub, BookWyrm federates with other instances and services like Mastodon and Pleroma, fostering a connected but independent network.
*   **Build Community and Connect:** Join or create communities focused on specific genres, interests, or friend groups.
*   **Enhanced Privacy & Moderation:** Control who sees your posts and manage your connections and federation.
*   **Collaborative Book Database:** BookWyrm helps build a decentralized database of books and authors through shared metadata.

## About BookWyrm

BookWyrm is a platform for social reading, designed to let you track what you're reading, review books, and follow your friends. While it includes basic cataloging functionalities, its primary focus is on social interaction and community.

## Federation

BookWyrm utilizes ActivityPub to enable federation with other BookWyrm instances and compatible services like Mastodon. This allows users to engage with each other across different platforms and create independent communities.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)

## Technology Stack

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
*   Flower Celery monitoring
*   Nginx HTTP server

## Set Up BookWyrm

Detailed instructions for setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production environment](https://docs.joinbookwyrm.com/install-prod.html) can be found in the [documentation website](https://docs.joinbookwyrm.com/).

## Contributing

You can contribute to BookWyrm in many ways. Learn how to get involved in the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
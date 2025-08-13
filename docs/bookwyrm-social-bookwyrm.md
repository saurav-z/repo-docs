# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, connect with fellow bookworms, and discuss your favorite reads.** (See the [original repo](https://github.com/bookwyrm-social/bookwyrm) for more details.)

[![GitHub release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Post reviews, share quotes, and discuss books with other readers.
*   **Reading Tracking:** Keep a record of what you're reading, have read, and want to read.
*   **Federation with ActivityPub:** Connect with users on other BookWyrm instances and ActivityPub-compliant platforms like Mastodon and Pleroma, building a decentralized network of book lovers.
*   **Privacy and Moderation:** Control your posts' visibility and choose which instances to federate with.
*   **Community Focused:** Join or create your own book-focused communities.

## About BookWyrm

BookWyrm is a social platform designed for readers. It offers features for tracking your reading, reviewing books, and connecting with friends. BookWyrm also includes basic book cataloging.

## Federation Explained

BookWyrm uses ActivityPub to connect different instances and other compatible services, like Mastodon. This design allows for smaller, community-driven platforms instead of large, centralized ones. For example, you can run an instance for your book club and still follow your friend's posts on a server dedicated to a specific genre.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [BookWyrm on Mastodon](https://tech.lgbt/@bookwyrm)

## Technology Stack

**Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub for federation
*   Celery task queuing
*   Redis task backend and activity stream manager

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Setting Up BookWyrm

Detailed installation instructions are available in the [documentation](https://docs.joinbookwyrm.com/), covering both development and production environments.

## Contributing

There are many ways you can help the BookWyrm project! Learn how to get involved in [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
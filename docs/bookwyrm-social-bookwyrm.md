# BookWyrm: Your Social Network for Books 📚

**BookWyrm is a decentralized social network that lets you connect with other readers, share reviews, and discover new books, all while maintaining control over your data.**  Find out more on the [original repo](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading & Reviews:** Share your thoughts, write reviews, and discuss books with other users in a social feed.
*   **Track Your Reading:** Easily keep track of books you've read, are currently reading, and want to read in the future.
*   **ActivityPub Federation:** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma.
*   **Decentralized Communities:** Join or create small, self-governed communities focused on specific interests or groups.
*   **Privacy & Moderation:** Customize your privacy settings and control which instances you federate with.

## About BookWyrm

BookWyrm provides a platform for social reading, allowing users to track their reading progress, review books, and follow friends.  It emphasizes social interaction and community building, offering an alternative to centralized platforms. BookWyrm leverages ActivityPub for federation, enabling interoperability with other services.

## Federation

BookWyrm is built on the ActivityPub protocol, enabling it to connect with other instances of BookWyrm and other compatible services like Mastodon. This allows you to participate in a decentralized network of readers.  You can create or join communities with specific interests and control the content you see.

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
*   Flower celery monitoring
*   Nginx HTTP server

## Get Started

Detailed instructions for setting up BookWyrm are available in the [documentation](https://docs.joinbookwyrm.com/). You can set up a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or a [production](https://docs.joinbookwyrm.com/install-prod.html) instance.

## Contributing

Want to help make BookWyrm better?  Learn how to contribute by visiting [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Links

*   [Project homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [@bookwyrm on Mastodon](https://tech.lgbt/@bookwyrm)
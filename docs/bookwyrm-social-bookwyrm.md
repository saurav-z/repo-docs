# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that empowers book lovers to connect, share reviews, and track their reading in a privacy-focused environment.** ([View the original repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm offers a unique social reading experience that goes beyond simply cataloging books. Built on the principles of federation, it allows users to join independent, trusted communities while still connecting with a wider network of book enthusiasts.

## Key Features:

*   **Share Your Reading:** Write reviews, discuss books, and share favorite quotes.
*   **Track Your Books:** Maintain a reading log of books you've read, are currently reading, and want to read.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and compatible platforms like Mastodon and Pleroma. Build a distributed database of books collaboratively.
*   **Privacy & Moderation:** Control post visibility and customize federation settings for a secure and personalized experience.
*   **Independent Communities:** Join or create your own BookWyrm instance for a self-determining, moderated community.

## About BookWyrm

BookWyrm prioritizes social interaction and community-building around books. While it provides basic cataloging functionalities, its main focus is on fostering discussion, sharing recommendations, and connecting with fellow readers.

## Federation: Connecting the Bookish World

BookWyrm leverages ActivityPub, an open standard that enables it to interoperate with other services like Mastodon. This means:

*   **Cross-Platform Interaction:** You can engage with users on different BookWyrm instances and other ActivityPub platforms.
*   **Community-Driven:** Run your own instance focused on a specific genre, book club, or group of friends. Each community has the autonomy to choose their federation partners and manage their members.
*   **Decentralized Data:** Federation helps build a shared, decentralized database of books and authors.

## Technology Stack

**Web Backend:**

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

## Get Started with BookWyrm

Detailed installation instructions for both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production setups](https://docs.joinbookwyrm.com/install-prod.html) are available on the [documentation website](https://docs.joinbookwyrm.com/).

## Contribute to BookWyrm

There are many ways to contribute to BookWyrm! Check out the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file to learn more about how you can get involved, even without coding experience.

## Useful Links

*   [Project homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Follow](https://tech.lgbt/@bookwyrm)
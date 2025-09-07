# BookWyrm: The Social Network for Bookworms

**BookWyrm is a decentralized social network where you can track your reading, discuss books, write reviews, and discover new literary adventures.** You can find the original project here: [BookWyrm on GitHub](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features:

*   **Social Reading:** Share your thoughts on books with reviews, quotes, and discussions.
*   **Reading Activity Tracking:** Maintain a list of books you've read, are reading, and want to read.
*   **Federation with ActivityPub:** Interact with users on other ActivityPub platforms like Mastodon and Pleroma, expanding your literary network.
*   **Decentralized Community:** Join or create self-managed instances for focused discussions and build trust-based connections.
*   **Privacy and Moderation:** Control your post visibility and manage your network through instance and user settings.

## What is BookWyrm?

BookWyrm is a social network designed for book lovers. It allows you to connect with other readers, share your reading experiences, and discover new books based on your interests and the recommendations of your friends. BookWyrm is built on the ActivityPub protocol, allowing it to federate with other platforms, such as Mastodon, and connect you to a vast community of readers.

## Federation and Community

BookWyrm utilizes ActivityPub to foster a decentralized and interconnected network. This allows you to:

*   **Connect with diverse communities:** Interact with users across different BookWyrm instances and other ActivityPub-compatible platforms.
*   **Build trusted networks:** Join or create instances tailored to specific interests or groups, fostering focused discussions and stronger connections.
*   **Maintain autonomy:** Each instance has control over its moderation and federation choices.

Learn more about ActivityPub implementation in BookWyrm by reading [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

**Backend:**

*   Django Web Server
*   PostgreSQL Database
*   ActivityPub Federation
*   Celery Task Queuing
*   Redis Task Backend & Activity Stream Manager

**Frontend:**

*   Django Templates
*   Bulma.io CSS Framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker-Compose
*   Gunicorn Web Runner
*   Flower Celery Monitoring
*   Nginx HTTP Server

## Get Started

Comprehensive setup instructions for both development and production environments are available in the [BookWyrm documentation](https://docs.joinbookwyrm.com/).

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Contribute

If you want to help this project out check out [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)
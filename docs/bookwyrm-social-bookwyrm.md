# BookWyrm: Your Social Network for Books

**BookWyrm is a decentralized social network built for readers to track their reading, connect with others, and discuss books, all while prioritizing community and privacy.**  Check out the original repository on GitHub: [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on what you're reading.
*   **Reading Tracking:** Maintain a list of books you've read, are reading, and want to read.
*   **Federation with ActivityPub:** Connect with users on other BookWyrm instances and compatible platforms like Mastodon and Pleroma.
*   **Community Focused:** Build or join communities with shared interests and values.
*   **Privacy & Moderation:** Control who sees your posts and manage your federated connections.

## Why Choose BookWyrm?

BookWyrm offers a refreshing alternative to centralized social media platforms, empowering readers to:

*   **Own Your Data:**  Your reading history and connections belong to you.
*   **Join a Community:** Connect with like-minded individuals in a safe and supportive environment.
*   **Support Decentralization:** Contribute to a more open and equitable internet.

## About BookWyrm

BookWyrm is a platform for social reading, enabling users to track their reading, review books, and connect with friends. It's built on the principles of open-source software and decentralized networking, offering a user-friendly experience for book lovers.

## Federation with ActivityPub

BookWyrm utilizes [ActivityPub](http://activitypub.rocks/) to interoperate with other instances of BookWyrm and with other ActivityPub-compliant services, such as Mastodon. This federation allows you to:

*   Interact with users across different platforms.
*   Share book metadata, collaboratively building a decentralized book database.
*   Create and participate in small, self-governing communities focused on specific interests.

Developers of other ActivityPub software can find out more about BookWyrm's implementation at [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

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

*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower Celery monitoring
*   Nginx HTTP server

## Get Started with BookWyrm

Find instructions on setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) on the [documentation website](https://docs.joinbookwyrm.com/).

## Contribute to BookWyrm

There are many ways to contribute to the BookWyrm project.  Learn how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Follow](https://tech.lgbt/@bookwyrm)
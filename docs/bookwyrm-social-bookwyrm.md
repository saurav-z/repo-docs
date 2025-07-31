# BookWyrm: A Social Network for Readers

**BookWyrm is a decentralized social network for book lovers, allowing you to track your reading, connect with friends, and discuss your favorite books in a federated and privacy-focused environment.** Learn more about the project at [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books with friends and the wider BookWyrm community.
*   **Reading Tracking:** Easily log what you're reading, what you've read, and what's on your to-read list.
*   **Federated Network:** Built on ActivityPub, BookWyrm integrates seamlessly with other platforms like Mastodon and Pleroma, fostering a decentralized and interconnected reading ecosystem.
*   **Privacy & Moderation:**  Control your posts' visibility and manage your instance's federation settings for a personalized experience.
*   **Discover Books:**  Explore a shared, decentralized database of books and authors, collaboratively built by the community.

## Why BookWyrm?

BookWyrm provides a refreshing alternative to centralized social networks, offering a space for readers to connect, share their passion for books, and build small, trusted communities. Unlike traditional platforms, BookWyrm promotes user agency and empowers you to curate your own online reading experience.

## Federation Explained

BookWyrm utilizes ActivityPub to connect different instances, allowing you to:

*   **Connect with friends across different BookWyrm servers.**
*   **Interact with users on other ActivityPub services like Mastodon and Pleroma.**
*   **Build and participate in niche communities focused on specific interests.**

This fosters a more intimate and controlled social experience, giving users more control over their content and connections.

## Technology Stack

**Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub for federation
*   Celery task queuing
*   Redis (for task backend and activity stream management)

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower for Celery monitoring
*   Nginx HTTP server

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)
*   **Federation Details:** [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md)
*   **Contributing:**  Find out how to contribute at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)

## Getting Started

The documentation website provides detailed instructions for setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.
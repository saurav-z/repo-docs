# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, review books, connect with friends, and discover new literature, all while maintaining control over your data.** ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features:

*   **Social Reading:** Share reviews, quotes, and discussions about books with other readers.
*   **Reading Tracking:** Organize your reading list, mark books as read, and track your progress.
*   **Federation with ActivityPub:** Interact with users and content on other ActivityPub platforms like Mastodon and Pleroma, and other BookWyrm instances.
*   **Decentralized Communities:** Join or create your own BookWyrm instance to connect with like-minded readers.
*   **Privacy & Moderation:** Control who sees your posts and what instances you federate with.

## About BookWyrm

BookWyrm is a social reading platform designed to provide a more user-centric and privacy-focused experience compared to centralized platforms. It's built on the ActivityPub protocol, enabling a federated network where users can connect across different instances and even interact with users on other compatible platforms like Mastodon. This allows for the creation of small, independent communities focused on shared interests.

## Federation

BookWyrm's ActivityPub implementation allows it to interoperate with other instances of BookWyrm and other ActivityPub services like Mastodon and Pleroma. This allows users to:

*   **Connect Across Platforms:** Interact with users on different instances of BookWyrm and other compatible services.
*   **Join Communities:** Create or join communities focused on specific interests.
*   **Control Your Data:** Manage your data within your chosen community.
*   **Collaborative Book Database:** Share metadata about books and authors.

Developers can find more details about BookWyrm's federation implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (task queue & activity stream)
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Get Started with BookWyrm

Detailed instructions for setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) are available on the documentation website.

## Contribute

Contribute to BookWyrm! Learn how to get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
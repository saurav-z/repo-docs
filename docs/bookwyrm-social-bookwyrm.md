# BookWyrm: The Social Network for Bookworms

**BookWyrm is a decentralized social network that empowers book lovers to connect, share reviews, and track their reading, fostering a vibrant community around literature.**

[![](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm allows you to track your reading progress, write book reviews, and discover new books to read, all within a community-driven social network.  BookWyrm uses ActivityPub for federation allowing you to connect with users on other BookWyrm instances and other ActivityPub-compatible platforms like Mastodon and Pleroma.

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books.
*   **Reading Tracking:** Maintain a reading list, track progress, and log your reading activity.
*   **Federation & Interoperability:** Connect with users on other BookWyrm instances and platforms like Mastodon and Pleroma via ActivityPub, building a decentralized database of books and authors.
*   **Community & Discovery:** Engage in discussions with fellow readers and discover new books based on your network's recommendations.
*   **Privacy Controls:** Manage who can see your posts and customize your federation with other instances.

## How BookWyrm Works

BookWyrm is a social network built on ActivityPub, allowing it to integrate with various social media platforms like Mastodon.  This allows for small, self-governed communities focused on specific interests, similar to those promoted by Run Your Own Social.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)
*   [BookWyrm GitHub Repository](https://github.com/bookwyrm-social/bookwyrm)

## Tech Stack

*   **Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Get Started

The [documentation website](https://docs.joinbookwyrm.com/) provides comprehensive instructions for setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contribute

Join the BookWyrm community and help make it even better! Learn how to contribute at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
# BookWyrm: A Decentralized Social Network for Book Lovers

**BookWyrm is the social network for readers who want to share reviews, track their reading, and connect with others in a federated, community-driven environment.** ([View the original repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm is a social platform built for book lovers. It allows you to track your reading progress, write and share reviews, discover new books, and connect with a community of readers. Unlike centralized platforms, BookWyrm uses federation, allowing users to join independent communities that can communicate and share content with each other, as well as other ActivityPub services like Mastodon and Pleroma.

## Key Features:

*   **Share Book Reviews and Recommendations:** Post your thoughts on books, recommend titles to friends, and engage in conversations about literature.
*   **Track Your Reading Journey:** Easily log books you've read, are currently reading, and want to read in the future.
*   **Federated Network:** Interact with users on other BookWyrm instances and ActivityPub-compatible services like Mastodon, expanding your reading community.
*   **Decentralized Database:** Collaborate with other instances to build and share a decentralized, community-maintained database of books and authors.
*   **Privacy & Control:** Users and administrators can manage their privacy settings and choose which instances to federate with.

## How BookWyrm Works:

BookWyrm leverages the ActivityPub protocol for federation, enabling interaction between different instances. This allows users to join smaller, independent communities focused on specific interests or groups of friends, offering a more personalized and controlled social reading experience.  This federation allows users on different instances to interact, share metadata, and build a decentralized database of books.

## Tech Stack:

BookWyrm utilizes a robust tech stack for its functionality:

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Getting Started:

Detailed instructions on setting up BookWyrm in a development or production environment are available in the [documentation](https://docs.joinbookwyrm.com/).

## Resources:

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Federation Details:** [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md)
*   **Mastodon:**  [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Contributing:

Interested in contributing to BookWyrm?  Learn how you can get involved by checking out the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file.
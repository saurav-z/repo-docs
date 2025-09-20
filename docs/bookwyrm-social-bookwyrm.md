# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized, open-source social network designed for book lovers to connect, share reviews, track reading progress, and discover new books.** ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, discuss books, and post quotes with fellow readers.
*   **Reading Activity Tracking:** Keep a record of books read, want-to-read lists, and reading progress.
*   **Decentralized Federation:** Powered by ActivityPub, allowing interaction with other BookWyrm instances and platforms like Mastodon and Pleroma.
*   **Community Focus:** Join or create small, self-governing communities focused on shared interests.
*   **Privacy Controls:** Manage post visibility and choose which instances to federate with.
*   **Collaborative Book Database:**  Book and author metadata is shared across the network, collaboratively building a rich, decentralized book database.

## About BookWyrm

BookWyrm offers a fresh approach to social reading, moving away from centralized platforms. It is designed to track your reading, review books, follow your friends, and discover new books. BookWyrm is built on the ActivityPub protocol, which enables federation and interoperability with other ActivityPub-compliant services. This allows users to connect with diverse communities while maintaining control over their data and experience.

## Federation

BookWyrm's ActivityPub integration enables interaction with other BookWyrm instances and services such as Mastodon, allowing users to engage in discussions, share reviews, and follow their friends across different platforms.

## Tech Stack

*   **Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Federation Details](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md)
*   [Mastodon Profile](https://tech.lgbt/@bookwyrm)

## Get Started

The [documentation website](https://docs.joinbookwyrm.com/) provides instructions for setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contribute

Help improve BookWyrm!  Find out how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
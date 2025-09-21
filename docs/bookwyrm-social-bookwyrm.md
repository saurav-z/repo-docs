# BookWyrm: The Social Network for Book Lovers

**Tired of centralized social networks? BookWyrm is a federated social network that lets you connect with friends, discover new books, and share your reading journey.**  ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share Your Reading:** Post reviews, comments, and quotes to discuss books with other readers.
*   **Track Your Reading:** Maintain a reading log and wishlist to keep track of your literary adventures.
*   **Federated Network:** Interact with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma.
*   **Decentralized Community:** Join or create small, self-governed communities with control over moderation and federation.
*   **Privacy and Moderation:** Control who sees your posts and which instances you interact with.

## Federation Explained

BookWyrm utilizes [ActivityPub](http://activitypub.rocks/), allowing users to connect across different BookWyrm instances and other ActivityPub-compatible platforms like Mastodon. This federated approach empowers users to form independent communities, fostering trust and personalized experiences.  This also allows for a decentralized database of books and authors to be collaboratively built.

## Get Involved

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)
*   [Contributing](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)

## Tech Stack

**Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis

**Frontend:** Django templates, Bulma.io, Vanilla JavaScript

**Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Set Up BookWyrm

Detailed instructions for setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) are available in the documentation.
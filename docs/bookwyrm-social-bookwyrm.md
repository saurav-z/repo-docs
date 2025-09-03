# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that helps you connect with friends, track your reading, and discuss books in a federated and privacy-focused environment.** ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and reading updates with others.
*   **Reading Tracking:** Organize your reading list and keep track of what you've read.
*   **Federation:** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma.
*   **Decentralized Communities:** Join or create your own instance for a focused book club or community.
*   **Privacy and Moderation:** Control your visibility and choose which instances you federate with.

## Learn More

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## About BookWyrm

BookWyrm is a social platform designed for book lovers to connect, share, and discover new reads. It's built on the principles of federation, allowing for a decentralized network of communities, each with its own focus and moderation.  It's not designed as a primary book data source but does facilitate cataloging to some degree.

## Federation Explained

BookWyrm uses [ActivityPub](http://activitypub.rocks/) to connect with other instances. This means you can follow friends on other BookWyrm servers or interact with users on platforms like Mastodon. This federation creates a network of independent communities, empowering users with control over their social experience.

## Tech Stack

*   **Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Setting up BookWyrm

Find detailed setup instructions in the [documentation website](https://docs.joinbookwyrm.com/), with guides for both [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Join the BookWyrm community and contribute to its growth! Learn how at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
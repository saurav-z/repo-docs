# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that empowers book lovers to connect, share reviews, and discover new reads.** ([See the original repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Post reviews, share quotes, discuss books, and connect with other readers.
*   **Reading Tracking:** Keep a record of what you're reading, have read, and want to read.
*   **Federated Network:** Built on ActivityPub, BookWyrm seamlessly interacts with other instances and platforms like Mastodon and Pleroma.
*   **Community-Focused:**  Join or create self-governed book communities and connect with like-minded individuals.
*   **Privacy & Moderation:** Control your post visibility and manage federation with other instances.
*   **Decentralized Database:** Collaboratively builds a shared database of books and authors.

## Why Choose BookWyrm?

BookWyrm offers a refreshing alternative to centralized book-focused social networks, promoting community-driven discussions, user privacy, and open-source principles.  Create a more intimate and focused experience where you can connect with others, share reviews, and find new books to love.

## Federation Explained

BookWyrm utilizes [ActivityPub](http://activitypub.rocks/) to connect with other instances and services like Mastodon, enabling a decentralized network of book lovers. This federation allows for small, self-governed communities. Check out [FEDERATION.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md) for more details on BookWyrm's implementation.

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker-compose, Gunicorn, Flower, Nginx

## Getting Started

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Follow us on Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)

Detailed setup instructions are available in the [documentation website](https://docs.joinbookwyrm.com/).

## Contributing

We welcome contributions from everyone! Learn how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
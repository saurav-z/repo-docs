# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, review books, and connect with fellow bookworms.** Explore and join the BookWyrm community on [GitHub](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books with others in the BookWyrm network.
*   **Reading Tracker:** Keep a record of the books you've read, are currently reading, and plan to read.
*   **Federation with ActivityPub:** Connect with users on other BookWyrm instances and services like Mastodon and Pleroma, fostering a decentralized social experience.
*   **Privacy and Moderation:** Control your posts' visibility and choose which instances to federate with, empowering you to create your own community and moderate as needed.
*   **Discover Books:** Explore a decentralized database of books and authors.

## About BookWyrm

BookWyrm is built on the ActivityPub protocol, allowing it to interoperate with other services. BookWyrm is designed for social reading, allowing you to track what you're reading, review books, and follow your friends. Learn more about the philosophy and logistics behind small, high-trust social networks at [runyourown.social](https://runyourown.social/).

## Federation

BookWyrm uses [ActivityPub](http://activitypub.rocks/) to federate with other instances of BookWyrm and compatible services. This allows users to interact across different platforms and build a shared, decentralized database of books and authors.

## Links

*   **Project Homepage:** [joinbookwyrm.com](https://joinbookwyrm.com/)
*   **Support:** [Patreon](https://patreon.com/bookwyrm)
*   **Documentation:** [docs.joinbookwyrm.com](https://docs.joinbookwyrm.com/)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Getting Started

The [documentation](https://docs.joinbookwyrm.com/) provides instructions for setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contributing

Interested in contributing? Explore the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file to see how you can get involved in the BookWyrm project.
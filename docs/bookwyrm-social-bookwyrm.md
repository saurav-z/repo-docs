# BookWyrm: A Social Network for Readers

**BookWyrm is a decentralized social network where you can track your reading, share reviews, discover new books, and connect with fellow book lovers in a privacy-focused and federated environment.**  [Visit the original repository on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers you to build a vibrant reading community, fostering discussions and shared reading experiences.

## Key Features

*   **Share Book Reviews & Discussions:** Compose reviews, comment on what you're reading, and post quotes from books to engage in conversations with other readers.
*   **Track Reading Progress:** Keep a record of the books you've read, are currently reading, and plan to read.
*   **Federated Social Network:** Connect with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma, creating a decentralized network of book enthusiasts.
*   **Privacy & Moderation:**  Control your data and interactions with robust privacy settings and moderation tools.
*   **Discover New Books:** Explore a collaborative, decentralized database of books built through federation and user contributions.

## About BookWyrm

BookWyrm offers a unique social reading experience, focusing on community and connection.  It's built on the ActivityPub protocol, allowing for interoperability with other platforms and fostering small, self-governing communities.

## Federation

BookWyrm's foundation in ActivityPub enables it to connect with other instances of BookWyrm and services like Mastodon. This federation allows users to interact across different platforms and build communities around shared reading interests.  Learn more about ActivityPub at [activitypub.rocks](http://activitypub.rocks/).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Get Started

Detailed instructions for setting up BookWyrm can be found in the [documentation](https://docs.joinbookwyrm.com/), including guides for [developer environments](https://docs.joinbookwyrm.com/install-dev.html) and [production deployments](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute

Interested in contributing to BookWyrm? Check out the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file to learn how you can get involved!
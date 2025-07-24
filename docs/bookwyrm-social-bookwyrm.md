# BookWyrm: A Social Network for Book Lovers 📚

BookWyrm is a decentralized social network, empowering readers to connect, share their reading experiences, and discover new books in a privacy-focused and community-driven environment. ([View the original repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books with other users.
*   **Reading Tracking:** Keep a record of books you've read, want to read, and are currently reading.
*   **Federation with ActivityPub:** Connect with users on other BookWyrm instances and other ActivityPub services like Mastodon and Pleroma.
*   **Decentralized Communities:** Join or create small, self-governed communities focused on specific interests.
*   **Privacy Controls:** Manage who can see your posts and which instances you federate with.

## About BookWyrm

BookWyrm is a social platform built for book enthusiasts. It allows you to track your reading, review books, follow friends, and discover new reads. Built on the ActivityPub protocol, it fosters interoperability with other platforms, offering a more decentralized and community-focused experience compared to centralized alternatives.

## Federation: Connect and Share

BookWyrm's ActivityPub implementation enables seamless interaction with other instances and services. This federation capability allows you to:

*   Interact with users on other BookWyrm instances.
*   Connect with users on ActivityPub-compliant platforms like Mastodon.
*   Participate in decentralized communities and discussions.
*   Contribute to a shared, collaborative book and author database.

For more information about BookWyrm's ActivityPub implementation, see [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (for task queuing & activity stream)
*   **Frontend:** Django templates, Bulma.io (CSS framework), Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Get Started

Explore the [documentation](https://docs.joinbookwyrm.com/) for detailed instructions on setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or in [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute

Join the BookWyrm community! Learn how you can contribute to the project's success at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network that empowers book lovers to connect, share, and discover new reads through the power of federation.** ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Connect with Readers:** Share reviews, discuss books, and engage in conversations with fellow bookworms across the network.
*   **Track Your Reading Journey:** Keep a personal record of books you've read, and create a list of books you want to read.
*   **Federated Social Networking:**  Leverage ActivityPub to interact with users on other BookWyrm instances, Mastodon, and other compatible platforms for a truly decentralized experience.
*   **Privacy and Control:** Manage your posts' visibility and choose which instances to federate with, ensuring a personalized and secure experience.
*   **Decentralized Book Database:** Collaborate to build a decentralized database of books through federation, enhancing the reading experience for everyone.

## Why Choose BookWyrm?

BookWyrm provides a refreshing alternative to centralized platforms, enabling you to:

*   **Join Communities:** Create or join book clubs and connect with communities that share your interests.
*   **Control Your Data:** Own your data and experience a more private, user-centric social network.
*   **Support Open Source:** Contribute to a community-driven project dedicated to building a better social reading experience.

## How BookWyrm Works

BookWyrm is built on the ActivityPub protocol, allowing interoperability with other ActivityPub-compliant services like Mastodon and Pleroma. This federation approach fosters a network of interconnected communities, empowering users to create small, trusted groups and control their online experience.

## Tech Stack

**Web Backend:**

*   Django
*   PostgreSQL
*   ActivityPub
*   Celery
*   Redis (task backend & activity stream manager)

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn
*   Flower
*   Nginx

## Get Started

The [BookWyrm documentation](https://docs.joinbookwyrm.com/) provides comprehensive instructions for setting up BookWyrm, including:

*   [Developer environment setup](https://docs.joinbookwyrm.com/install-dev.html)
*   [Production installation guide](https://docs.joinbookwyrm.com/install-prod.html)

## Contribute

Help make BookWyrm even better! Learn how you can contribute to the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Additional Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support BookWyrm](https://patreon.com/bookwyrm)
*   [Follow BookWyrm on Mastodon](https://tech.lgbt/@bookwyrm)
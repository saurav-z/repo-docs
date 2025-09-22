# BookWyrm: A Social Network for Book Lovers 📚

**Tired of centralized social networks?** BookWyrm offers a decentralized, open-source platform for readers to connect, share reviews, and discover new books. ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Connect with other readers:** Discuss books, share reviews, and follow your friends within the BookWyrm network.
*   **Track Your Reading:** Easily manage your reading lists, mark books as read, and keep track of your progress.
*   **Federated Social Networking:** Interact with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon, fostering a decentralized and interconnected reading community.
*   **Privacy & Moderation:**  Control your post visibility and who you federate with, allowing you to create a safe and personalized reading experience.
*   **Post about Books:**  Compose reviews, share quotes, and discuss books with other users across the network.

## About BookWyrm

BookWyrm is designed to be a social platform for readers, allowing users to track their reading, review books, and connect with others. It leverages the power of ActivityPub to offer a federated experience.  This allows for the creation of small, independent communities where users can build and manage their own spaces.

## Federation & ActivityPub

BookWyrm is built on the ActivityPub protocol, which allows it to interoperate with other ActivityPub services like Mastodon and Pleroma.  This enables:

*   **Cross-Platform Interaction:** Users can engage with each other regardless of the instance or platform they use.
*   **Decentralized Data:**  Metadata about books and authors is shared across the network, contributing to a collaborative, decentralized book database.

## Links

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Tech Stack

BookWyrm is built using a modern tech stack:

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis, and Redis (for activity streams)
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, and Nginx

## Setting up BookWyrm

Detailed installation instructions for both developer and production environments are available in the [documentation](https://docs.joinbookwyrm.com/).

## Contributing

Want to help make BookWyrm even better? Check out the [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) file to learn how you can get involved!
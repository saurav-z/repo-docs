# BookWyrm: Your Decentralized Social Network for Readers

**BookWyrm is a social platform that empowers readers to connect, share, and discover books in a privacy-focused, federated environment.**  ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:**  Share reviews, quotes, and thoughts on books with other readers.
*   **Reading Activity Tracking:** Keep a record of what you've read, what you're reading, and your reading wish list.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and compatible platforms like Mastodon, fostering a decentralized network.
*   **Privacy and Moderation:** Control your posts' visibility and manage your federated connections.
*   **Discover New Books:** Explore a collaborative, decentralized database of books and authors.

##  Why BookWyrm?

BookWyrm provides a refreshing alternative to centralized social networks, offering:

*   **Community-Focused Experience:** Join or create book clubs and communities centered around shared interests.
*   **Data Ownership:**  Maintain control over your data and interactions.
*   **Interoperability:**  Connect with readers across the Fediverse, expanding your social network.

##  About BookWyrm

BookWyrm is built on the ActivityPub protocol, enabling interoperability with other services like Mastodon and Pleroma. This allows for small, self-governed communities and a more open and decentralized social experience. Find out more about the philosophy and logistics behind small, high-trust social networks at [https://runyourown.social/](https://runyourown.social/).

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [BookWyrm on Mastodon](https://tech.lgbt/@bookwyrm)

## Tech Stack

**Web Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend and activity stream manager

**Front End:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Set Up BookWyrm

Detailed instructions for setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) are available in the documentation.

## Contributing

Learn how you can contribute to BookWyrm's development and community at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
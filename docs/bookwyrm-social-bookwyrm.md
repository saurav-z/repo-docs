# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized, social network built for readers to connect, share reviews, and discover new books.** ([Original Repository](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers readers to engage in a social reading experience, allowing you to track your reading, share your thoughts, and discover what to read next. It's built on ActivityPub, allowing federation with other BookWyrm instances, and services like Mastodon and Pleroma.

## Key Features:

*   **Share & Discuss Books:** Write and share book reviews, post quotes, and discuss books with other readers.
*   **Track Your Reading:** Keep a record of books you've read, are currently reading, and want to read in the future.
*   **Decentralized Social Network:** Built on ActivityPub for federation, enabling interaction with users on other instances and services.
*   **Community-Driven:** Join or create communities, connect with like-minded readers, and build your own online reading experience.
*   **Privacy & Moderation:** Control your privacy settings and manage your interactions within the network.
*   **Collaborative Book Database:** BookWyrm collaboratively builds a decentralized database of books and authors.

## Benefits of Using BookWyrm:

*   **Connect with Friends & Like-Minded Readers:** Build your own online reading community, connect with friends, and engage in meaningful discussions.
*   **Discover New Books:** Explore recommendations from your network and discover books you might have missed.
*   **Support Open Source:** Contribute to a project dedicated to building a community-owned platform for readers.
*   **Avoid Algorithmic Control:** You control your data.  Enjoy a timeline free of algorithmic influence.
*   **Join the Fediverse:** Interact with other users on ActivityPub compliant services, such as Mastodon and Pleroma.

## Links

*   [Project homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon](https://tech.lgbt/@bookwyrm)

## Technology Stack:

### Web Backend
*   Django
*   PostgreSQL
*   ActivityPub
*   Celery
*   Redis (Task Backend)
*   Redis (Activity Stream Manager)

### Front End
*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

### Deployment
*   Docker and docker-compose
*   Gunicorn
*   Flower
*   Nginx

## Get Started:

Detailed installation instructions are available in the [documentation](https://docs.joinbookwyrm.com/). This includes guides for setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or a [production environment](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute:

BookWyrm is an open-source project. Learn how you can contribute to the project's growth at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
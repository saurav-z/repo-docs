# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, connect with other bookworms, and discover your next favorite read.** Check out the original repo at [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm).

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm is a social network for readers that allows users to track reading progress, write reviews, and interact with others. It is built on the ActivityPub protocol and supports federation, letting users join small, trusted communities that connect with each other.

## Key Features:

*   **Social Reading:** Share your reading experiences, write reviews, comment on books, and post quotes.
*   **Reading Tracking:** Keep a record of the books you've read and your "to-read" list.
*   **ActivityPub Federation:** Connect with users on other BookWyrm instances and other ActivityPub-compliant services like Mastodon, expanding your social reading network.
*   **Privacy and Moderation:** Control who can see your posts and manage your interactions on the network.
*   **Decentralized Communities:** Join or create communities focused on specific interests or groups, fostering trust and autonomy.

## Links

[![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

-   [Project homepage](https://joinbookwyrm.com/)
-   [Support](https://patreon.com/bookwyrm)
-   [Documentation](https://docs.joinbookwyrm.com/)

## Technology Stack:

**Backend:**

*   Django
*   PostgreSQL
*   ActivityPub
*   Celery
*   Redis (for task and activity stream management)

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn
*   Flower (Celery monitoring)
*   Nginx

## Set up BookWyrm

Visit the [documentation website](https://docs.joinbookwyrm.com/) to learn how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Learn how you can contribute to the BookWyrm project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
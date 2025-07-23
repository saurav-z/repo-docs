# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network where you can track your reading, share reviews, and connect with fellow bookworms.** This platform allows you to engage with books, build a reading community, and discover new literary treasures. For more details, check out the original repository: [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share Book Reviews and Recommendations:** Write reviews, comment on what you're reading, and post quotes, fostering discussions with other book lovers.
*   **Track Your Reading Progress:** Keep a record of the books you've read, and create a wishlist of books you want to read in the future.
*   **Federated Social Network:** Interact with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma, building a decentralized network.
*   **Privacy & Moderation:** Customize your privacy settings and control which instances you federate with, enabling you to build your community and moderate your interactions.
*   **Discover New Reads:** Discover new books and authors through shared reviews, recommendations, and discussions.

## About BookWyrm

BookWyrm is more than just a book cataloging system. It's a social platform designed for readers to connect.  It is built on ActivityPub, which enables it to interoperate with different instances of BookWyrm, as well as other ActivityPub compliant services, like Mastodon.

## Federation: Connect and Collaborate

BookWyrm utilizes ActivityPub to connect with other BookWyrm instances and compatible services, fostering small, self-determined communities. This enables users to create and join reading groups focused on specific interests, or build a network among friends. You can choose which instances to federate with and manage your community autonomously.

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis.
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript.
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx.

## Get Started with BookWyrm

Explore the [documentation website](https://docs.joinbookwyrm.com/) for detailed instructions on setting up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or in [production](https://docs.joinbookwyrm.com/install-prod.html).

## Useful Links

*   [Project homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Follow](https://tech.lgbt/@bookwyrm)

## Contribute to BookWyrm

Discover how you can contribute to the project by checking out [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
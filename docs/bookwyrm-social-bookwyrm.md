# BookWyrm: The Social Network for Bookworms

**BookWyrm is a decentralized social network for book lovers to connect, share reviews, and discover new reads.** ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books with other readers.
*   **Reading Tracking:** Keep a record of books you've read, are currently reading, and want to read.
*   **Decentralized Network (Federation):** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma.
*   **Community-Focused:** Join or create book clubs and small, trusted communities with control over moderation and federation.
*   **Privacy & Moderation:** Customize your privacy settings and control your interactions.
*   **Discover New Books:** Explore recommendations and discover what others are reading.

## Why Choose BookWyrm?

BookWyrm offers a refreshing alternative to centralized book social networks by prioritizing user control, community, and privacy. It uses ActivityPub to connect users across instances, creating a decentralized network that fosters genuine interaction and a richer reading experience.

## Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Follow](https://tech.lgbt/@bookwyrm)

## About BookWyrm

BookWyrm is a platform designed for social reading, enabling users to track their reading, review books, and connect with friends. It's built on the ActivityPub protocol, allowing interoperability with other ActivityPub-compliant services. BookWyrm emphasizes community, user control, and building a decentralized database of books through collaboration.

## Federation with ActivityPub

BookWyrm utilizes the ActivityPub protocol to connect with other instances of BookWyrm and other platforms like Mastodon. This enables users to interact across different servers, creating a decentralized network where users can form and participate in a variety of communities, such as those based on interest or friendship.

Developers can learn more about BookWyrm's ActivityPub implementation at [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (task queuing & activity stream)
*   **Front End:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Setting Up BookWyrm

Detailed instructions for setting up BookWyrm can be found in the [documentation website](https://docs.joinbookwyrm.com/), including setup in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production environment](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

There are many ways to contribute to BookWyrm, even without coding skills. Learn how you can get involved by visiting [`CONTRIBUTING.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
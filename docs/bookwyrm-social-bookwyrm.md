# BookWyrm: A Social Network for Book Lovers

BookWyrm is a decentralized social network, built for readers by readers, that lets you track your reading, connect with friends, and discover new books. Check out the original repository on GitHub: [bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Post reviews, discuss books, and share your thoughts with other readers.
*   **Reading Tracker:** Keep a record of what you're reading, have read, and want to read.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and other ActivityPub-compliant platforms like Mastodon and Pleroma.
*   **Decentralized Communities:** Join or create your own instance with like-minded readers, fostering a sense of community and control.
*   **Privacy and Moderation:** Control your post visibility and choose which instances you want to federate with.
*   **Discover New Books:** Browse a collaborative and decentralized book database.

## About BookWyrm

BookWyrm is designed as a social reading platform, allowing users to connect and share their love of books.  It's built on the ActivityPub protocol, enabling federation with other instances and platforms.  This approach fosters independent communities and gives users control over their social experience.

## Federation

BookWyrm utilizes the ActivityPub protocol to enable interoperability. This allows instances to connect and share information, including book metadata, leading to a collectively built, decentralized book database.  BookWyrm users can interact with users on different BookWyrm instances and other ActivityPub services such as Mastodon.

For developers, more details on BookWyrm's ActivityPub implementation can be found in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (task queue & activity stream)
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker and docker-compose, Gunicorn, Flower, Nginx

## Get Started

Visit the [documentation website](https://docs.joinbookwyrm.com/) for instructions on setting up BookWyrm in both a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

The BookWyrm project welcomes contributions!  Learn how you can get involved by checking out [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, connect with others, and discover new books, all while prioritizing community and privacy.** ([See the original repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers bookworms to connect, share reviews, and build a thriving, decentralized bookish community. Built on ActivityPub, BookWyrm integrates with other platforms like Mastodon, fostering a collaborative environment for readers everywhere.

## Key Features:

*   **Social Reading:** Post reviews, share quotes, and discuss books with other users.
*   **Reading Tracking:** Keep a record of books you've read, are currently reading, and want to read.
*   **Federation:** Interact with users on other ActivityPub-compliant platforms like Mastodon and Pleroma.
*   **Decentralized & Community-Focused:** Join or create small, self-governed communities with customized moderation and federation settings.
*   **Privacy & Moderation:** Users control post visibility and instance federation, allowing for a safe and customizable experience.

## BookWyrm in Detail

BookWyrm isn't just about cataloging books; it's about cultivating a social experience around reading. It leverages ActivityPub to connect with other BookWyrm instances and compatible services, allowing users to build connections and share their love of books across different platforms.

### Federation Explained

BookWyrm's strength lies in its federation capabilities.  Based on ActivityPub, it fosters interoperability with other BookWyrm instances, as well as with platforms like Mastodon.  This allows users to participate in a broader network while still enjoying the autonomy and intimacy of smaller, self-governed communities.

Learn more about BookWyrm's ActivityPub implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

BookWyrm uses a robust and modern tech stack.

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Front End:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Get Started

Explore the [Project homepage](https://joinbookwyrm.com/), and the [documentation website](https://docs.joinbookwyrm.com/) for installation instructions in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Support & Community

*   [Support](https://patreon.com/bookwyrm)
*   Follow BookWyrm on Mastodon:  [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Contributing

BookWyrm thrives on community contributions.  Learn how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
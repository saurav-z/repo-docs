# BookWyrm: The Decentralized Social Network for Book Lovers

**BookWyrm is a social network built for readers, by readers, fostering book discussions, reviews, and reading recommendations in a decentralized and privacy-focused environment.** ([See the original repository](https://github.com/bookwyrm-social/bookwyrm))

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Linting](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Post reviews, share quotes, and discuss books with other readers across the network.
*   **Reading Tracking:**  Keep a record of books you've read, are currently reading, and plan to read.
*   **Federation with ActivityPub:** Interact seamlessly with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon and Pleroma. Share book metadata and collaboratively build a decentralized book database.
*   **Privacy & Moderation:**  Control the visibility of your posts and manage your connections, fostering a safe and tailored reading experience.
*   **Community Focused:** Create or join communities that can be focused on a specific interest, for a group of friends, or anything else that brings people together.

## Benefits

*   **Decentralization:** Enjoy a social network free from corporate control, offering true ownership of your data and interactions.
*   **Community Building:** Connect with like-minded readers and build meaningful relationships around shared interests.
*   **Discover New Books:** Explore a wide range of books and authors through recommendations, reviews, and discussions.
*   **Privacy-Focused:** Take control of your data and online experience with robust privacy features.

## About BookWyrm

BookWyrm is a platform for social reading. You can use it to track what you're reading, review books, and follow your friends. It isn't primarily meant for cataloguing or as a data-source for books, but it does do both of those things to some degree.

## Federation

BookWyrm is built on [ActivityPub](http://activitypub.rocks/). With ActivityPub, it inter-operates with different instances of BookWyrm, and other ActivityPub compliant services, like Mastodon. This means you can run an instance for your book club, and still follow your friend who posts on a server devoted to 20th century Russian speculative fiction. It also means that your friend on mastodon can read and comment on a book review that you post on your BookWyrm instance.

Federation makes it possible to have small, self-determining communities, in contrast to the monolithic service you find on GoodReads or Twitter. An instance can be focused on a particular interest, be just for a group of friends, or anything else that brings people together. Each community can choose which other instances they want to federate with, and moderate and run their community autonomously. Check out https://runyourown.social/ to get a sense of the philosophy and logistics behind small, high-trust social networks.

Developers of other ActivityPub software can find out more about BookWyrm's implementation at [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Get Started

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Follow on Mastodon:**  [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)

## Set Up BookWyrm

The [documentation website](https://docs.joinbookwyrm.com/) has instructions on how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html).

## Contributing

Contribute to the BookWyrm project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md)
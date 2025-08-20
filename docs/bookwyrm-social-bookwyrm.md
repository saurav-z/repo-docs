# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you connect with other book lovers, share your thoughts, and discover your next favorite read.**  [Visit the original repo on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Share Your Reading:** Post reviews, quotes, and thoughts on books you're reading.
*   **Track Your Reading Journey:**  Keep a record of the books you've read and the ones you plan to read.
*   **Connect with Fellow Readers:** Engage in discussions with other BookWyrm users and build a community around books.
*   **Federation with ActivityPub:** Interact seamlessly with users on other ActivityPub platforms like Mastodon and Pleroma, expanding your social reach.
*   **Decentralized & Community-Focused:** Join or create a BookWyrm instance that aligns with your interests, fostering a sense of community and control.
*   **Privacy & Moderation:** Customize your privacy settings and moderate your own instance, promoting a safe and enjoyable experience.

## About BookWyrm

BookWyrm offers a unique platform for social reading, going beyond simple book cataloging.  It allows you to track your reading, review books, and connect with friends. BookWyrm is built on ActivityPub, enabling it to interact with various instances of BookWyrm and other ActivityPub-compliant services.

## Federation Explained

BookWyrm's foundation in [ActivityPub](http://activitypub.rocks/) allows it to seamlessly integrate with other instances and services, such as [Mastodon](https://joinmastodon.org/) and [Pleroma](http://pleroma.social/). This connectivity enables you to engage with users and share book-related content across different platforms, creating a decentralized and interconnected social reading ecosystem.

## Tech Stack

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (for task queuing and activity streams)
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript
*   **Deployment:** Docker, docker-compose, Gunicorn, Flower, Nginx

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Federation Implementation:** [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md)

## Getting Started

Detailed instructions on setting up BookWyrm are available in the documentation:  [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)

*   [Developer Setup](https://docs.joinbookwyrm.com/install-dev.html)
*   [Production Setup](https://docs.joinbookwyrm.com/install-prod.html)

## Contribute

Find out how to get involved with the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
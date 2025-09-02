# BookWyrm: A Social Network for Book Lovers 📚

BookWyrm is a decentralized, social network that empowers book lovers to connect, share reviews, and discover new reads within a federated, community-driven environment.  **(See the original repository on GitHub: [bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm))**

[![GitHub release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading & Reviews:**  Share reviews, quotes, and discuss books with other BookWyrm users.
*   **Reading Tracking:** Keep a personal record of books you've read and want to read.
*   **Federated Network (ActivityPub):** Interact with users on other BookWyrm instances, and other ActivityPub services like Mastodon and Pleroma, for a more open and connected experience.
*   **Decentralized Community:**  Join or create smaller, self-governed communities with greater control over content and interaction.
*   **Privacy & Moderation:**  Control your privacy settings and choose which instances to interact with.

## About BookWyrm

BookWyrm is built on the ActivityPub protocol, allowing users to connect with other instances of BookWyrm and other compatible platforms, such as Mastodon and Pleroma. This fosters a decentralized network of communities where users can engage in thoughtful discussions and share their passion for reading. BookWyrm is designed for social reading, offering features to track your reading, review books, and follow friends, while also contributing to a collaborative, decentralized database of books and authors.

## Federation Explained

BookWyrm uses ActivityPub to connect different instances and other services. This means you can follow friends on other servers and they can engage with your content. It facilitates independent communities with the ability to choose their own moderation and community standards, in contrast to larger, centralized platforms. Learn more about running your own social network at [https://runyourown.social/](https://runyourown.social/).

## Tech Stack

*   **Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis
*   **Frontend:** Django Templates, Bulma.io, Vanilla JavaScript
*   **Deployment:** Docker, Docker-compose, Gunicorn, Flower, Nginx

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Get Involved

Learn how to set up BookWyrm for [development](https://docs.joinbookwyrm.com/install-dev.html) or [production](https://docs.joinbookwyrm.com/install-prod.html) in the documentation.  Contribute to the project by visiting [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
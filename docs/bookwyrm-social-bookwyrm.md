# BookWyrm: A Social Network for Readers

**BookWyrm is a decentralized social network that lets you connect with other readers, share reviews, and discover new books.** [Visit the original repository on GitHub](https://github.com/bookwyrm-social/bookwyrm)

[![Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers you to track your reading, discuss books, write reviews, and discover new titles within a federated, community-driven environment. Built on the ActivityPub protocol, BookWyrm seamlessly integrates with other platforms like Mastodon, fostering a decentralized social reading experience.

## Key Features

*   **Social Reading & Reviews:** Post reviews, comments, and quotes, engaging in conversations with fellow readers across the network.
*   **Reading Activity Tracking:** Maintain a personal reading log, tracking what you've read, are currently reading, and plan to read.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and compatible platforms, expanding your reading community and building a shared book database.
*   **Privacy & Moderation:** Control your post visibility and manage your instance's federation settings for a secure and tailored social experience.
*   **Community Building:** Join or create book clubs, connect with like-minded readers, and build your own trusted reading community.

## About BookWyrm

BookWyrm focuses on facilitating social interaction around books. While it catalogs books and authors, its primary function is to enable users to connect, share, and discuss their reading experiences. This includes tracking reading progress, creating reviews, and following friends.

## Federation: Connect and Collaborate

BookWyrm utilizes ActivityPub to provide interoperability with various instances of BookWyrm and other ActivityPub-compatible services such as Mastodon and Pleroma. This allows you to:

*   Follow friends across different platforms.
*   Share and engage with reviews on any compatible server.
*   Join small, self-governed communities with a focus on shared interests.

## Tech Stack

**Backend:**

*   Django (Web Server)
*   PostgreSQL (Database)
*   ActivityPub (Federation)
*   Celery (Task Queuing)
*   Redis (Task Backend & Activity Stream Manager)

**Frontend:**

*   Django Templates
*   Bulma.io (CSS Framework)
*   Vanilla JavaScript

**Deployment:**

*   Docker & Docker-compose
*   Gunicorn (Web Runner)
*   Flower (Celery Monitoring)
*   Nginx (HTTP Server)

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)

## Get Started with BookWyrm

Detailed instructions on how to set up BookWyrm for both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments can be found in the documentation.

## Contribute

Help shape the future of BookWyrm! Learn how to contribute to the project by visiting [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
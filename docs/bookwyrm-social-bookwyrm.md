# BookWyrm: Your Decentralized Social Network for Book Lovers

**BookWyrm is a powerful, open-source social network designed to connect book enthusiasts, enabling them to track their reading, share reviews, and discover new books in a federated and privacy-focused environment.**  Discover more at the [BookWyrm GitHub Repository](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features of BookWyrm

*   **Social Reading & Book Discovery:** Share reviews, discuss books, and connect with fellow readers.
*   **Reading Activity Tracking:** Easily track what you're reading, have read, and want to read.
*   **Federated Network:** Connect with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma.
*   **Privacy & Moderation:** Control your data and who sees your posts, with options for instance moderation.
*   **Open-Source & Community-Driven:** Contribute to and shape the future of social reading.

## Why Choose BookWyrm?

BookWyrm empowers you to:

*   **Build meaningful connections:** Engage in discussions with readers who share your passion for books.
*   **Enjoy a decentralized experience:**  Join or create your own community, free from the constraints of centralized platforms.
*   **Prioritize privacy and control:**  Manage your data and customize your experience to your preferences.
*   **Contribute to a collaborative ecosystem:** Help build a decentralized, open-source platform for book lovers.

## How Federation Works

BookWyrm utilizes ActivityPub, enabling seamless interaction with other ActivityPub-compliant platforms. This allows you to:

*   Follow friends on Mastodon and other platforms.
*   Have your reviews and posts read by users on other instances.
*   Contribute to a collectively built, decentralized book database.

## Technical Details

BookWyrm is built using a robust tech stack:

*   **Backend:** Django (web server), PostgreSQL (database), ActivityPub (federation), Celery (task queuing), Redis (task backend & activity stream manager)
*   **Frontend:** Django templates, Bulma.io (CSS framework), Vanilla JavaScript
*   **Deployment:** Docker, Docker Compose, Gunicorn, Flower, Nginx

## Get Started with BookWyrm

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:** [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

**Set up Instructions:**  The [documentation website](https://docs.joinbookwyrm.com/) provides detailed instructions for setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contribute to BookWyrm

We welcome contributions from everyone!  Learn how you can help at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
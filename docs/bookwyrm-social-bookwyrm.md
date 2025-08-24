# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network where you can track your reading, connect with other book enthusiasts, and discover your next favorite read.** (See the original repo: [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm is built on the ActivityPub protocol, allowing you to join a federated network of independent instances and interact with users on platforms like Mastodon and Pleroma. This means you can create or join a community around your specific interests and control your data.

## Key Features

*   **Social Reading:** Post reviews, discuss books, share quotes, and converse with other bookworms.
*   **Reading Tracking:** Keep track of what you're reading, have read, and plan to read.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and ActivityPub-compatible services (like Mastodon), fostering a decentralized community.
*   **Decentralized Database of Books:** BookWyrm collaboratively builds a decentralized database of books and authors via federation.
*   **Privacy and Moderation:** Control your posts' visibility and the instances you federate with.
*   **Community-driven:** Run your own instance for a book club, friends, or any interest!

## Resources

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Mastodon:**  [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Technical Details

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis.
*   **Frontend:** Django templates, Bulma.io CSS framework, Vanilla JavaScript.
*   **Deployment:** Docker, Docker-compose, Gunicorn, Flower, Nginx.

## Getting Started

Explore the comprehensive documentation at [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/) for detailed instructions on setting up BookWyrm in a development or production environment.

## Contributing

We welcome contributions! Learn how to get involved in the BookWyrm project by visiting [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
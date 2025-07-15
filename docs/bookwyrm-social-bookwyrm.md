# BookWyrm: The Social Network for Book Lovers

**Tired of centralized social media platforms? BookWyrm is a decentralized, open-source social network designed for readers to connect, share reviews, and track their reading journey, fostering a community built on the principles of privacy and federation.**  You can find the original repository [here](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Connect and Share:** Post book reviews, share quotes, and discuss books with fellow readers across the network.
*   **Track Your Reading:** Maintain a reading log to record books you've read, want to read, and are currently reading.
*   **Federation with ActivityPub:** Interact with users on other ActivityPub platforms like Mastodon and Pleroma, fostering a truly decentralized social experience.
*   **Build a Decentralized Book Database:**  Share book and author metadata to collectively build a rich, open book database.
*   **Privacy and Control:** Manage your post visibility and control which instances you federate with, ensuring a safe and personalized experience.
*   **Community Focused:** Build or join smaller, trusted communities, fostering meaningful connections.

## BookWyrm: Your Open-Source Social Reading Platform

BookWyrm is built on ActivityPub and is designed to provide a space for readers to track their reading, discuss books, and discover new ones.  It's an alternative to centralized platforms, allowing for a more community-focused, privacy-respecting experience.

## Federation Explained

BookWyrm utilizes ActivityPub to connect different instances of BookWyrm, and other ActivityPub services like Mastodon. This federation allows you to:

*   Run your own instance for a book club or a group of friends.
*   Follow and interact with users on different servers.
*   Share and discuss reviews across platforms.
*   Build a decentralized network of book metadata.

## Technology Stack

**Web Backend:**

*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend and activity stream manager

**Front End:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and docker-compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [Mastodon Follow](https://tech.lgbt/@bookwyrm)

## Getting Started

The [documentation website](https://docs.joinbookwyrm.com/) provides detailed instructions on setting up BookWyrm in both [developer](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contribute

Learn how you can contribute to the BookWyrm project by exploring [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
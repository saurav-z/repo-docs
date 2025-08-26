# BookWyrm: The Social Network for Book Lovers 

**BookWyrm is a decentralized social network that lets you connect with fellow readers, share reviews, and discover new books.**  [View the original repository](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub release (latest by date)](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm is a social network designed specifically for book lovers. Track your reading, write reviews, and discover new books while connecting with others in a decentralized, federated environment. Unlike centralized platforms, BookWyrm empowers users to build and join communities with shared interests, all while maintaining control over their data and experience.

## Key Features

*   **Share Book Reviews and Recommendations:** Post reviews, share quotes, and discuss books with other BookWyrm users.
*   **Track Your Reading:** Easily log books you've read, are currently reading, and want to read in the future.
*   **Federated Social Network (ActivityPub):** Interact with users on other BookWyrm instances and compatible ActivityPub services like Mastodon and Pleroma. Build a decentralized book database through shared metadata.
*   **Privacy and Moderation Controls:** Manage your posts' visibility and choose which instances to federate with, fostering a safe and customized social experience.
*   **Discover New Books and Authors:** Explore shared reading lists and recommendations from your network.

## How BookWyrm Works: Federation and Community

BookWyrm utilizes ActivityPub, an open, decentralized social networking protocol. This means:

*   **Connect with anyone:** Interact with users on other BookWyrm instances and compatible platforms like Mastodon.
*   **Create your own community:** Run an instance for your book club or interest group.
*   **Control your experience:** Choose your instance and who you interact with.

## Tech Stack

**Web Backend:**

*   Django (Web Server)
*   PostgreSQL (Database)
*   ActivityPub (Federation)
*   Celery (Task Queuing)
*   Redis (Task Backend & Activity Stream Manager)

**Front End:**

*   Django Templates
*   Bulma.io (CSS Framework)
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn (Web Runner)
*   Flower (Celery Monitoring)
*   Nginx (HTTP Server)

## Get Started with BookWyrm

Visit the [official documentation](https://docs.joinbookwyrm.com/) to learn how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or for [production](https://docs.joinbookwyrm.com/install-prod.html).

## Community and Contribution

BookWyrm thrives on community contributions! Explore how you can get involved at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md) and help shape the future of social reading.
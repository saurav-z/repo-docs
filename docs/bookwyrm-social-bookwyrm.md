# BookWyrm: A Social Network for Book Lovers (Decentralized Reading)

BookWyrm is a revolutionary social platform for book lovers, allowing you to track your reading, share reviews, and connect with other readers in a decentralized and privacy-focused environment. For the original project, visit [the BookWyrm GitHub repository](https://github.com/bookwyrm-social/bookwyrm).

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading:** Share your thoughts on books with reviews, quotes, and discussions.
*   **Reading Tracking:** Keep a record of the books you've read, are currently reading, and want to read.
*   **Decentralized Network (Federation):** Connect with users on other BookWyrm instances and other ActivityPub services like Mastodon and Pleroma, fostering a community-driven ecosystem.
*   **Privacy & Control:** Manage your posts' visibility and choose which instances to federate with.
*   **Community Focus:** Build and join communities centered around specific interests or groups of friends.
*   **Collaborative Book Database:** Participate in building a decentralized database of books and authors through federation.

## Why BookWyrm?

Unlike centralized platforms, BookWyrm promotes smaller, self-governed communities.  This allows for greater control over your data and content, promoting a more intimate and engaging reading experience.

## Federation:  Connecting Communities

BookWyrm utilizes the ActivityPub protocol to communicate with other instances of BookWyrm and other ActivityPub-compatible services.  This allows you to:

*   Interact with users on platforms like Mastodon and Pleroma.
*   Join or create your own BookWyrm instance, tailored to your specific interests (e.g., a book club).
*   Maintain control over your online experience within a smaller, more trusted community.

## Technology Stack

**Web Backend:**

*   Django
*   PostgreSQL
*   ActivityPub
*   Celery
*   Redis (for task backend and activity stream)

**Frontend:**

*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**

*   Docker and Docker Compose
*   Gunicorn
*   Flower (Celery monitoring)
*   Nginx

## Get Started

Learn how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or a [production environment](https://docs.joinbookwyrm.com/install-prod.html) by visiting the [official documentation](https://docs.joinbookwyrm.com/).

## Contribute

Contribute to the BookWyrm project!  Find out more at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
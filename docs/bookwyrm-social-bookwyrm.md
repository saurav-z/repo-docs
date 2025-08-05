# BookWyrm: The Social Network for Book Lovers

**BookWyrm is a decentralized social network that lets you track your reading, discuss books, and connect with other bookworms in a privacy-focused environment.** [(View the project on GitHub)](https://github.com/bookwyrm-social/bookwyrm)

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm empowers you to build and participate in a thriving community of readers, offering a refreshing alternative to centralized platforms.  It's built on ActivityPub, offering robust federation to connect you to other compatible platforms like Mastodon.

## Key Features of BookWyrm:

*   **Social Reading:** Share reviews, quotes, and thoughts on books with friends and the wider BookWyrm community.
*   **Reading Tracking:** Maintain a personal library and keep track of your reading progress, books you've read, and those you want to read.
*   **Federation with ActivityPub:** Interact with users on other BookWyrm instances and compatible platforms, building a decentralized social network and a shared book metadata database.
*   **Privacy & Moderation:** Control your visibility and curate your experience by choosing who can see your posts and which instances to federate with.
*   **Community Focused:** Build and participate in self-determining communities focused on your specific interests.

## About BookWyrm

BookWyrm is designed for social interaction centered around reading.  It allows you to track your reading, write reviews, and follow your friends while also providing some basic cataloging features.  It prioritizes community and control, with federation being a key feature.

## Federation: Connect and Collaborate

BookWyrm is built on ActivityPub, enabling it to interact with other BookWyrm instances and ActivityPub-compliant services like Mastodon and Pleroma. This federation model allows for small, independent communities, giving users more control over their online experience. You can run your own BookWyrm instance for a book club, connect with friends across different instances, and share your reading experiences.

Developers can learn more about BookWyrm's ActivityPub implementation in [`FEDERATION.md`](https://github.com/bookwyrm-social/bookwyrm/blob/main/FEDERATION.md).

## Tech Stack

BookWyrm utilizes a robust tech stack to provide a seamless user experience:

**Backend:**
*   Django web server
*   PostgreSQL database
*   ActivityPub federation
*   Celery task queuing
*   Redis task backend and activity stream manager

**Frontend:**
*   Django templates
*   Bulma.io CSS framework
*   Vanilla JavaScript

**Deployment:**
*   Docker and docker-compose
*   Gunicorn web runner
*   Flower celery monitoring
*   Nginx HTTP server

## Get Started with BookWyrm

Find detailed installation instructions for both development and production environments on the [BookWyrm documentation website](https://docs.joinbookwyrm.com/).

## Contribute to BookWyrm

Help improve BookWyrm!  Learn how to contribute to the project at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).

## Useful Links

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)
*   [@BookWyrm on Mastodon](https://tech.lgbt/@bookwyrm)
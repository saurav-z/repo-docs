# BookWyrm: A Social Network for Book Lovers

**Tired of centralized social networks? BookWyrm is a decentralized social network built for bookworms, enabling you to share your reading experiences, connect with friends, and discover new books.** ([Original Repo](https://github.com/bookwyrm-social/bookwyrm))

[![GitHub Release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

## Key Features

*   **Social Reading & Reviews:** Post reviews, comment on books, share quotes, and engage in discussions with other readers.
*   **Reading Tracking:** Keep track of your reading progress, create "to-read" lists, and log books you've completed.
*   **Decentralized Federation:** Built on ActivityPub, BookWyrm allows you to connect with other BookWyrm instances and compatible platforms like Mastodon and Pleroma, fostering a decentralized social reading experience.
*   **Privacy & Control:** Manage your privacy settings and choose which instances you federate with, creating a personalized and safe environment.
*   **Discover New Books:** Discover new books and authors through the collective knowledge of the BookWyrm network.

## Why Choose BookWyrm?

BookWyrm empowers you to build and join small, trusted communities focused on reading and book discussions. Its decentralized nature means you control your data and your social experience, free from the constraints of centralized platforms.

## Federation Explained

BookWyrm's use of ActivityPub allows for a seamless connection with other ActivityPub-compatible services, creating a unified social web. This allows for:

*   Interaction between users across different BookWyrm instances.
*   Integration with platforms such as Mastodon.
*   A collaborative, decentralized database of books and authors.

This federation approach promotes user autonomy and allows for specialized communities, letting you find the perfect space for your reading interests.

## Resources

*   [Project Homepage](https://joinbookwyrm.com/)
*   [Support](https://patreon.com/bookwyrm)
*   [Documentation](https://docs.joinbookwyrm.com/)

## Technology Stack

**Backend:**

*   Django
*   PostgreSQL
*   ActivityPub
*   Celery
*   Redis (for task queuing and activity stream management)

**Frontend:**

*   Django templates
*   Bulma.io CSS Framework
*   Vanilla JavaScript

**Deployment:**

*   Docker & Docker Compose
*   Gunicorn
*   Flower
*   Nginx

## Get Started

Find instructions on how to set up BookWyrm in a [developer environment](https://docs.joinbookwyrm.com/install-dev.html) or [production environment](https://docs.joinbookwyrm.com/install-prod.html).

## Contribute

Help improve BookWyrm! Learn more about contributing at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
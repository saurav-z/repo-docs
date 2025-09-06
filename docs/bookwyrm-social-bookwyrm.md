# BookWyrm: A Social Network for Book Lovers

**BookWyrm is a decentralized social network designed for readers to connect, share reviews, and discover new books.** (See the original repo here: [https://github.com/bookwyrm-social/bookwyrm](https://github.com/bookwyrm-social/bookwyrm)).

[![GitHub release](https://img.shields.io/github/release/bookwyrm-social/bookwyrm.svg?colorB=58839b)](https://github.com/bookwyrm-social/bookwyrm/releases)
[![Run Python Tests](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/django-tests.yml)
[![Pylint](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml/badge.svg)](https://github.com/bookwyrm-social/bookwyrm/actions/workflows/pylint.yml)

BookWyrm offers a unique space for readers to engage with each other, track their reading progress, and build a community centered around a love of books. It's built on ActivityPub, allowing for federation and interoperability with other platforms like Mastodon and Pleroma, fostering a decentralized and community-driven experience.

## Key Features

*   **Social Reading:** Share reviews, quotes, and thoughts on books, and engage in discussions with other readers.
*   **Reading Tracking:** Keep a record of the books you've read, are currently reading, and want to read.
*   **Decentralized Network (Federation):** Connect with users on other BookWyrm instances and ActivityPub-compatible platforms like Mastodon, promoting a more open and community-driven social experience.
*   **Privacy & Moderation:** Control your visibility and who you interact with, enabling a customizable and safe user experience.
*   **Book Discovery:** Discover new books and authors through the shared reading activity of your network.

## How BookWyrm Works

BookWyrm is built on the ActivityPub protocol, enabling federation and interoperability with other platforms, like Mastodon. This means you can run your own instance or join an existing one, and still connect with users on other servers. This decentralized approach allows for small, focused communities and fosters autonomy, offering a refreshing alternative to monolithic social networks.

## Links

*   **Project Homepage:** [https://joinbookwyrm.com/](https://joinbookwyrm.com/)
*   **Support:** [https://patreon.com/bookwyrm](https://patreon.com/bookwyrm)
*   **Documentation:** [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/)
*   **Mastodon:**  [![Mastodon Follow](https://img.shields.io/mastodon/follow/000146121?domain=https%3A%2F%2Ftech.lgbt&style=social)](https://tech.lgbt/@bookwyrm)

## Tech Stack

BookWyrm utilizes a robust tech stack to provide a reliable and scalable platform:

*   **Web Backend:** Django, PostgreSQL, ActivityPub, Celery, Redis (for task queuing and activity stream), Gunicorn, Flower, Nginx
*   **Front End:** Django templates, Bulma.io (CSS framework), Vanilla JavaScript

## Get Started

Detailed instructions for setting up BookWyrm can be found on the official documentation website: [https://docs.joinbookwyrm.com/](https://docs.joinbookwyrm.com/). Installation guides are available for both [development](https://docs.joinbookwyrm.com/install-dev.html) and [production](https://docs.joinbookwyrm.com/install-prod.html) environments.

## Contributing

BookWyrm is an open-source project and welcomes contributions from everyone. Learn how to get involved and help shape the future of BookWyrm at [CONTRIBUTING.md](https://github.com/bookwyrm-social/bookwyrm/blob/main/CONTRIBUTING.md).
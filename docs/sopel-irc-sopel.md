# Sopel: Your Easy-to-Use Python IRC Bot

**Sopel** is a lightweight, open-source IRC bot written in Python, designed for simplicity, ease of use, and extensibility. Build your own IRC bot with custom plugins to manage your channel and more.  Get started today with [the original Sopel repository](https://github.com/sopel-irc/sopel)!

[![PyPI version](https://img.shields.io/pypi/v/sopel.svg)](https://pypi.python.org/pypi/sopel)
[![Build Status](https://github.com/sopel-irc/sopel/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/sopel-irc/sopel/actions/workflows/ci.yml?query=branch%3Amaster+event%3Apush)
[![GitHub issues](https://img.shields.io/github/issues/sopel-irc/sopel.svg)](https://github.com/sopel-irc/sopel/issues)
[![Coverage Status](https://coveralls.io/repos/github/sopel-irc/sopel/badge.svg?branch=master)](https://coveralls.io/github/sopel-irc/sopel?branch=master)
[![License](https://img.shields.io/pypi/l/sopel.svg)](https://github.com/sopel-irc/sopel/blob/master/COPYING)

## Key Features

*   **Simple and Lightweight:** Easy to set up, run, and manage.
*   **Open Source:** Freely available and customizable.
*   **Extensible with Plugins:** Easily add new functionality.
*   **Database Support:** Supports SQLite, MySQL, PostgreSQL, MSSQL, Oracle, Firebird, and Sybase for persistent data.
*   **Python-based:**  Leverages the power and flexibility of Python.

## Installation

### Latest Stable Release

The easiest way to install Sopel is using pip:

```bash
pip install sopel
```

### From Source

1.  Clone the repository or download a source archive:

    ```bash
    git clone https://github.com/sopel-irc/sopel.git
    ```

2.  Navigate to the source directory and install:

    ```bash
    cd sopel
    pip install -e .
    ```

3.  Run to start the bot:

    ```bash
    sopel
    ```

   *   **Note:** Sopel requires Python 3.8 or higher.

## Database Support

Sopel uses SQLAlchemy to provide support for various databases, including:

*   SQLite
*   MySQL
*   PostgreSQL
*   MSSQL
*   Oracle
*   Firebird
*   Sybase

Configure database settings using the following options: `db_type`, `db_filename` (SQLite only), `db_driver`, `db_user`, `db_pass`, `db_host`, `db_port`, and `db_name`.  You'll need to install the necessary packages for your chosen database.

*   **Important Note:** Plugins created before Sopel 7.0 *may* have issues with databases other than SQLite.

## Adding Plugins

*   Place custom plugins in the `~/.sopel/plugins` directory.
*   Search PyPI for installable plugins.
*   Explore the [sopel-extras](https://github.com/sopel-irc/sopel-extras) repository for additional plugins.
*   Learn to create your own plugins with the [plugin tutorial](https://sopel.chat/tutorials/part-1-writing-plugins/).

## Further Documentation

Find detailed information on the official website:

*   [Official Website](https://sopel.chat/)
*   [Built-in Commands](https://sopel.chat/usage/commands/)
*   [Tutorials](https://sopel.chat/tutorials/)
*   [API Documentation](https://sopel.chat/docs/)
*   [Usage Information](https://sopel.chat/usage/)

## Get Involved

Have questions or need help? Join us on IRC in `#sopel <irc://irc.libera.chat/#sopel>`_ on Libera Chat.

## Donate

Support the Sopel project:

*   [Sponsor Sopel on GitHub](https://github.com/sponsors/sopel-irc)
*   [Donate via Open Collective](https://opencollective.com/sopel)

Donations help cover infrastructure costs. Transparency for expenses can be found on our Open Collective profile.
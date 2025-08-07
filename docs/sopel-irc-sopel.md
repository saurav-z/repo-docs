# Sopel: Your Lightweight and Extensible IRC Bot in Python

**Sopel is an easy-to-use, open-source IRC bot written in Python, designed for extensibility and customization.**

[View the original repository on GitHub](https://github.com/sopel-irc/sopel)

## Key Features

*   **Easy to Use and Run:** Sopel is designed for simplicity, making it easy to set up and manage your IRC bot.
*   **Extensible:**  Customize your bot with plugins, supporting a wide variety of functions and commands.
*   **Lightweight:** Sopel's design ensures it runs efficiently, using minimal resources.
*   **Database Support:** Leverages SQLAlchemy to support multiple database types (SQLite, MySQL, PostgreSQL, MSSQL, Oracle, Firebird, and Sybase) for data persistence.
*   **Plugin Ecosystem:**  Easily add functionality through a vibrant ecosystem of plugins, with a built-in system for plugin installation.
*   **Comprehensive Documentation:**  Benefit from an official website with detailed usage instructions, tutorials, API documentation, and a list of built-in commands.

## Installation

### Latest Stable Release

The recommended way to install Sopel is using pip:

```bash
pip install sopel
```

### Other Installation Methods

*   **Arch Linux:** Install the `sopel` package from the [community] repository.
*   **From Source:** Download the latest source archive or clone the repository from GitHub and run `pip install -e .` within the source directory.  Requires Python 3.8+.

## Adding Plugins

*   Place new plugins in the `~/.sopel/plugins` directory.
*   Install newer plugins as packages via PyPI.
*   Explore the `sopel-extras` repository and other community resources to find existing plugins.
*   Create your own plugins with the help of the official [plugin tutorial](https://sopel.chat/tutorials/part-1-writing-plugins/).

## Database Configuration

Sopel utilizes SQLAlchemy and supports a variety of databases. Configure database options in your configuration file including: `db_type`, `db_filename` (SQLite only), `db_driver`, `db_user`, `db_pass`, `db_host`, `db_port`, and `db_name`.

## Further Documentation

*   **Official Website:** [https://sopel.chat/](https://sopel.chat/)
    *   Commands: [https://sopel.chat/usage/commands/](https://sopel.chat/usage/commands/)
    *   Tutorials: [https://sopel.chat/tutorials/](https://sopel.chat/tutorials/)
    *   API Documentation: [https://sopel.chat/docs/](https://sopel.chat/docs/)
    *   Usage Information: [https://sopel.chat/usage/](https://sopel.chat/usage/)

## Get Help

*   Join us in `#sopel <irc://irc.libera.chat/#sopel>`_ on Libera Chat.

## Support Sopel

*   [Sponsor Sopel](https://github.com/sponsors/sopel-irc) on GitHub.
*   Donate through [Open Collective](https://opencollective.com/sopel).
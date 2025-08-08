# Sopel: The Simple, Lightweight, and Extensible IRC Bot

Sopel is a powerful and user-friendly IRC bot written in Python, designed to be easy to use, run, and extend.

[View the original repository on GitHub](https://github.com/sopel-irc/sopel)

## Key Features

*   **Easy to Use:**  Sopel is designed with simplicity in mind, making it easy for both beginners and experienced users to set up and manage.
*   **Lightweight:**  Efficient design ensures minimal resource consumption, ideal for various hosting environments.
*   **Extensible:**  Create or use existing plugins to customize Sopel's functionality, adding commands, features, and integrations.
*   **Open Source:**  Benefit from community contributions, transparency, and the freedom to modify and redistribute the software.
*   **Database Support:** Supports SQLite, MySQL, PostgreSQL, MSSQL, Oracle, Firebird, and Sybase, using SQLAlchemy.
*   **Python-based:** Leverage the power and flexibility of the Python programming language.

## Installation

### Latest Stable Release

The recommended way to install Sopel is using `pip`:

```bash
pip install sopel
```

### From Source

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sopel-irc/sopel.git
    ```
    Or download a source archive from GitHub.

2.  **Navigate to the source directory:**

    ```bash
    cd sopel
    ```

3.  **Install:**

    ```bash
    pip install -e .
    ```

4.  **Run the bot:**

    ```bash
    sopel
    ```

    *Note: Requires Python 3.8+.*

## Database Support

Sopel utilizes SQLAlchemy, offering support for various database types: SQLite (default), MySQL, PostgreSQL, MSSQL, Oracle, Firebird, and Sybase.

To configure alternative databases, use the following options: `db_type`, `db_filename` (SQLite only), `db_driver`, `db_user`, `db_pass`, `db_host`, `db_port`, and `db_name`.  You may need to install required database drivers separately.

## Adding Plugins

1.  **Default Plugin Directory:** Place your plugins in the `~/.sopel/plugins` directory.
2.  **Installable Packages:** Search PyPI for available Sopel plugins: [PyPI Search](https://pypi.org/search/?q=%22sopel%22)
3.  **Sopel-Extras Repository:**  Explore additional plugins in the `sopel-extras <https://github.com/sopel-irc/sopel-extras>`_ repository.
4.  **Create Your Own:**  A `tutorial <https://sopel.chat/tutorials/part-1-writing-plugins/>`_ is available for creating new plugins.

## Further Information

*   **Official Website:**  [https://sopel.chat/](https://sopel.chat/)
    *   Built-in `commands <https://sopel.chat/usage/commands/>`_
    *   `Tutorials <https://sopel.chat/tutorials/>`_
    *   `API documentation <https://sopel.chat/docs/>`_
    *   `Usage information <https://sopel.chat/usage/>`_

## Get Help

*   **Join the Community:** `#sopel <irc://irc.libera.chat/#sopel>`_ on Libera Chat.

## Donations

Support the Sopel project:

*   **GitHub Sponsors:** [https://github.com/sponsors/sopel-irc](https://github.com/sponsors/sopel-irc)
*   **Open Collective:** [https://opencollective.com/sopel](https://opencollective.com/sopel)

Donations are used for infrastructure costs.  Transparency is maintained through Open Collective expenses.
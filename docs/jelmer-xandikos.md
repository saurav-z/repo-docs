# Xandikos: Your Lightweight, Git-Backed CalDAV/CardDAV Server

**Tired of complex server setups? Xandikos provides a simple yet powerful way to manage your calendars and contacts, all backed by the reliability of Git.**  [See the original repo](https://github.com/jelmer/xandikos).

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

Xandikos (Ξανδικός) takes its name from the month of March in the ancient Macedonian calendar.

**Key Features:**

*   **Lightweight & Efficient:** Designed for ease of use and minimal resource consumption.
*   **Git-Backed:** Leverages Git for version control, data integrity, and easy backups.
*   **CalDAV & CardDAV Compliant:** Fully supports the latest standards for calendar and contact synchronization.
*   **Docker Support:** Easily deploy and manage Xandikos with Docker containers.
*   **Flexible Configuration:** Customize your setup with various command-line options and environment variables.

**Implemented Standards:**

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - Partially implemented
*   RFC 3744 (Access Control) - Partially implemented
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

**Limitations:**

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

**Supported Clients:**

Xandikos works seamlessly with a wide range of CalDAV/CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   sogo connector for Icedove/Thunderbird
*   aCALdav syncer for Android
*   pycardsyncer
*   akonadi
*   CalDAV-Sync
*   CardDAV-Sync
*   Calendarsync
*   Tasks
*   AgendaV
*   CardBook
*   Apple's iOS
*   Home Assistant's CalDAV integration
*   pimsync
*   davcli
*   Thunderbird

**Dependencies:**

Xandikos requires:

*   Python 3 (see pyproject.toml for specific version)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

**Installation (Example - Debian/Ubuntu):**

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

Alternatively, you can install dependencies using pip:

```bash
python setup.py develop
```

**Docker:**

A Dockerfile is provided for easy deployment. Regularly built images are available at ``ghcr.io/jelmer/xandikos``. See the `Container overview page <https://github.com/jelmer/xandikos/pkgs/container/xandikos>`_ for a full list of available tags.

**Configuration via environment variables:**

*   ``PORT`` - Port to listen on (default: 8000)
*   ``METRICS_PORT`` - Port for metrics endpoint (default: 8001)
*   ``LISTEN_ADDRESS`` - Address to bind to (default: 0.0.0.0)
*   ``DATA_DIR`` - Data directory path (default: /data)
*   ``CURRENT_USER_PRINCIPAL`` - User principal path (default: /user/)
*   ``ROUTE_PREFIX`` - URL route prefix (default: /)
*   ``AUTOCREATE`` - Auto-create directories (true/false)
*   ``DEFAULTS`` - Create default calendar/addressbook (true/false)
*   ``DEBUG`` - Enable debug logging (true/false)
*   ``DUMP_DAV_XML`` - Print DAV XML requests/responses (true/false)
*   ``NO_STRICT`` - Enable client compatibility workarounds (true/false)

See `examples/docker-compose.yml` and the `man page <https://www.xandikos.org/manpage.html>`_ for more details.

**Running:**

1.  **Standalone (for testing):**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a server listening on `http://localhost:8080`.

2.  **Production:** Run Xandikos behind a reverse proxy like Apache or nginx. See the `examples/` directory for example configurations.

**Client Instructions:**

Clients that support RFC:`5397` can automatically discover the calendars and addressbook URLs from the server, providing the base URL to Xandikos during setup.

Clients that lack such automated discovery need the full URL to your calendar or addressbook. If you initialized Xandikos with the ``--defaults`` argument, the URLs will look something like this:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

**Contributing:**

Contributions are welcome! Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new) and read `CONTRIBUTING.md`.

**Help & Support:**

*   IRC: *#xandikos* on the `OFTC <https://www.oftc.net/>`_ network.
*   Mailing List: `Xandikos <https://groups.google.com/forum/#!forum/xandikos>`_
*   Extended documentation:  `on the home page <https://www.xandikos.org/docs/>`_.
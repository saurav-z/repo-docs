# Xandikos: A Lightweight, Git-Backed CalDAV/CardDAV Server

**Xandikos provides a powerful and flexible CalDAV and CardDAV server that leverages the versioning capabilities of Git for data storage.** ([See the original repo](https://github.com/jelmer/xandikos))

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

Xandikos, named after the Macedonian month, offers a robust solution for managing your calendars and contacts with the added benefit of Git's version control.  Find extended documentation on the [official documentation site](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Based Storage:** All data is stored and versioned within a Git repository.
*   **CalDAV and CardDAV Compliance:**  Fully implements core RFCs for seamless integration with a variety of clients.
*   **Lightweight and Efficient:** Designed for performance and ease of use.
*   **Docker Support:** Easy deployment with pre-built Docker images.
*   **Flexible Configuration:**  Configurable through command-line arguments and environment variables.

## Implemented Standards

Xandikos supports a wide range of WebDAV, CalDAV, and CardDAV standards:

*   **Core WebDAV:** RFC 4918/2518 (Implemented, excluding LOCK operations)
*   **CalDAV:** RFC 4791 (Fully Implemented)
*   **CardDAV:** RFC 6352 (Fully Implemented)
*   **Current Principal:** RFC 5397 (Fully Implemented)
*   **Versioning Extensions:** RFC 3253 (Partially implemented)
*   **Access Control:** RFC 3744 (Partially implemented)
*   **POST to create members:** RFC 5995 (Fully Implemented)
*   **Extended MKCOL:** RFC 5689 (Fully Implemented)
*   **Collection Synchronization for WebDAV:** RFC 6578 (Fully Implemented)
*   **Calendar Availability:** RFC 7953 (Fully Implemented)

For more details on specification compliance, see the [DAV compliance notes](notes/dav-compliance.rst).

## Limitations

*   **No Multi-User Support:** Xandikos is designed for single-user deployments.
*   **No CalDAV Scheduling Extensions:** Scheduling features are not currently implemented.

## Supported Clients

Xandikos has been tested and works with the following CalDAV/CardDAV clients:

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
*   homeassistant's CalDAV integration
*   pimsync
*   davcli
*   Thunderbird
*   ...and more!

## Dependencies

Xandikos requires:

*   Python 3 (see pyproject.toml for specific version)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install Dependencies:

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```
OR
```bash
python setup.py develop
```

## Docker

Xandikos provides a Dockerfile for easy deployment. The Docker image is regularly built and published at `ghcr.io/jelmer/xandikos`.  Use the `v$RELEASE` tags (e.g., `v0.2.11`) for specific releases.  View the [Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos) for a full list.

**Environment Variables for Docker Configuration:**

*   `PORT`: Port to listen on (default: 8000)
*   `METRICS_PORT`: Port for metrics endpoint (default: 8001)
*   `LISTEN_ADDRESS`: Address to bind to (default: 0.0.0.0)
*   `DATA_DIR`: Data directory path (default: /data)
*   `CURRENT_USER_PRINCIPAL`: User principal path (default: /user/)
*   `ROUTE_PREFIX`: URL route prefix (default: /)
*   `AUTOCREATE`: Auto-create directories (true/false)
*   `DEFAULTS`: Create default calendar/addressbook (true/false)
*   `DEBUG`: Enable debug logging (true/false)
*   `DUMP_DAV_XML`: Print DAV XML requests/responses (true/false)
*   `NO_STRICT`: Enable client compatibility workarounds (true/false)

See the `examples/docker-compose.yml` file and the [man page](https://www.xandikos.org/manpage.html) for detailed configuration information.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.

**Running a Standalone Instance (for testing):**

```bash
./bin/xandikos --defaults -d $HOME/dav
```
This will start a server on `http://localhost:8080/`.

**Production Deployment:**

For production, it's recommended to run Xandikos behind a reverse proxy like Apache or nginx.  See the `examples/` directory for init system configuration examples.

## Client Instructions

Some clients automatically discover calendar and addressbook URLs (RFC 5397 support). For those, simply provide the base URL.  For clients without autodiscovery, use these URLs:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Please report bugs and request features on [GitHub](https://github.com/jelmer/xandikos/issues/new).  If you'd like to contribute code or documentation, read the [CONTRIBUTING](CONTRIBUTING.md) guide.  Look for the `new-contributor` label on issues for beginner-friendly tasks.

## Help

*   IRC: `#xandikos` on the OFTC IRC network
*   Mailing List: [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
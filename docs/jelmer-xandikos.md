# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a self-hosted CalDAV and CardDAV server that leverages the power of Git for data storage, offering a secure and version-controlled solution for managing your calendars and contacts.**  [View the original repository](https://github.com/jelmer/xandikos)

![Xandikos Logo](logo.png)
_Find extended documentation [on the home page](https://www.xandikos.org/docs/)._

## Key Features

*   **Git-Backed Storage:**  Your calendar and contact data is stored in a Git repository, providing version control, easy backups, and data integrity.
*   **Standards Compliant:** Implements a wide range of CalDAV and CardDAV standards, ensuring compatibility with a variety of clients.
*   **Lightweight & Efficient:** Designed to be resource-friendly, making it suitable for various hosting environments.
*   **Docker Support:**  Easily deployable with a provided Dockerfile and pre-built images on GitHub Container Registry.
*   **Flexible Configuration:** Configure Xandikos with environment variables for Docker or through command-line options for standalone use.

## Implemented Standards

Xandikos supports a broad range of WebDAV, CalDAV, and CardDAV standards:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - *Partial*
*   RFC 3744 (Access Control) - *Partial*
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos is compatible with a wide array of CalDAV and CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   Sogo Connector for Icedove/Thunderbird
*   aCALdav syncer for Android
*   pycardsyncer
*   Akonadi
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

## Dependencies

Xandikos is written in Python 3 (see `pyproject.toml` for specifics) and uses:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker Deployment

A Dockerfile is provided for easy deployment. The Docker image is available at `ghcr.io/jelmer/xandikos`.

**Configuration:** Use environment variables to configure the Docker image. Key variables include:

*   `PORT`
*   `METRICS_PORT`
*   `LISTEN_ADDRESS`
*   `DATA_DIR`
*   `CURRENT_USER_PRINCIPAL`
*   `ROUTE_PREFIX`
*   `AUTOCREATE`
*   `DEFAULTS`
*   `DEBUG`
*   `DUMP_DAV_XML`
*   `NO_STRICT`

See `examples/docker-compose.yml` and the `man page <https://www.xandikos.org/manpage.html>`_ for detailed configuration options.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.

**Standalone Example:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a server listening on `localhost:8080`, with a default calendar and address book.

## Client Instructions

Most clients can auto-discover URLs. For clients that require specific URLs, use the following format:

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome!  Report bugs and suggest features via [GitHub issues](https://github.com/jelmer/xandikos/issues/new).  See `CONTRIBUTING.md` for contribution guidelines.  Issues tagged `new-contributor` are suitable for newcomers.

## Help

*   IRC:  *#xandikos* on the OFTC IRC network.
*   Mailing List:  [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
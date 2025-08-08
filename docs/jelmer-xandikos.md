# Xandikos: Your Lightweight Git-Backed CardDAV/CalDAV Server

**Xandikos** is a powerful, yet easy-to-use, CardDAV and CalDAV server that stores your calendar and contact data in a Git repository, providing robust version control and data integrity. [**View on GitHub**](https://github.com/jelmer/xandikos)

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

For extended documentation, visit the [Xandikos Documentation](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Backed Storage:** Leverages Git for version control, backups, and data integrity.
*   **Complete Standard Compliance:** Implements a wide range of CardDAV and CalDAV standards.
*   **Lightweight and Efficient:** Designed for performance and ease of deployment.
*   **Docker Support:** Easily deployable using Docker containers.
*   **Flexible Configuration:** Customizable through environment variables and command-line options.
*   **Client Compatibility:** Works with a variety of popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos offers robust support for essential WebDAV, CalDAV, and CardDAV standards:

*   **Core WebDAV (RFC 4918/2518):** Implemented (excluding LOCK operations - COPY/MOVE implemented for non-collections)
*   **CalDAV (RFC 4791):** Fully implemented.
*   **CardDAV (RFC 6352):** Fully implemented.
*   **Current Principal (RFC 5397):** Fully implemented.
*   **Versioning Extensions (RFC 3253):** Partially implemented.
*   **Access Control (RFC 3744):** Partially implemented.
*   **POST to create members (RFC 5995):** Fully implemented.
*   **Extended MKCOL (RFC 5689):** Fully implemented.
*   **Collection Synchronization for WebDAV (RFC 6578):** Fully implemented.
*   **Calendar Availability (RFC 7953):** Fully implemented.

For detailed compliance information, see [DAV Compliance](notes/dav-compliance.rst).

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos has been tested and works seamlessly with a wide range of CalDAV and CardDAV clients:

*   Vdirsyncer
*   caldavzap/carddavmate
*   evolution
*   DAVx5
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

## Dependencies

Xandikos is built using Python 3 (see `pyproject.toml` for specific versions) and PyPy 3. It utilizes the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

### Installation of Dependencies
Install dependencies using pip or your system's package manager.  For Debian/Ubuntu:

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

or via pip:

```bash
python setup.py develop
```

## Docker Deployment

Xandikos provides a Dockerfile for easy deployment.  The image is regularly built and published on [ghcr.io/jelmer/xandikos](https://github.com/jelmer/xandikos/pkgs/container/xandikos). Use the `v$RELEASE` tags, such as `v0.2.11`.

### Docker Configuration

Configure the Docker image using these environment variables:

*   `PORT` (default: 8000)
*   `METRICS_PORT` (default: 8001)
*   `LISTEN_ADDRESS` (default: 0.0.0.0)
*   `DATA_DIR` (default: /data)
*   `CURRENT_USER_PRINCIPAL` (default: /user/)
*   `ROUTE_PREFIX` (default: /)
*   `AUTOCREATE` (true/false)
*   `DEFAULTS` (true/false)
*   `DEBUG` (true/false)
*   `DUMP_DAV_XML` (true/false)
*   `NO_STRICT` (true/false)

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for advanced configuration.

## Running Xandikos

Xandikos can run directly on HTTP or behind a reverse proxy like Apache or Nginx.

### Testing

Run a standalone instance (no authentication) with a pre-created calendar/addressbook:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access your server at `http://localhost:8080`.

Xandikos doesn't create collections without `--defaults`. Create collections using your CalDAV/CardDAV client or by creating git repositories in the *contacts* or *calendars* directories.

### Production

Deploy Xandikos behind a reverse proxy such as Apache or nginx. See examples/ for init system configurations.

## Client Instructions

Some clients can auto-discover calendar and addressbook URLs (if they support RFC 5397). For these, enter the base URL to Xandikos.  Clients that require the specific URLs may resemble this:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! File issues for bugs or feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new). Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines. Issues for new contributors are tagged with `new-contributor <https://github.com/jelmer/xandikos/labels/new-contributor>`.

## Help

Get help through the `#xandikos` IRC channel on [OFTC](https://www.oftc.net/) or the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
# Xandikos: Your Lightweight CardDAV/CalDAV Server with Git Backing

**Xandikos is a powerful and simple CalDAV/CardDAV server that uses a Git repository for data storage.**  Get started with this easy-to-use server and manage your calendars and contacts with the power of Git! 

[View the original repository on GitHub](https://github.com/jelmer/xandikos)

![Xandikos Logo](logo.png)

*   **Lightweight and Efficient:** Xandikos is designed for simplicity and performance.
*   **Git-Backed Storage:** Leverage the versioning, backup, and collaboration features of Git for your calendar and contact data.
*   **Standards Compliant:** Implements core WebDAV, CalDAV, CardDAV, and other relevant RFCs.
*   **Docker Support:** Easily deploy and manage Xandikos using Docker containers.

## Key Features

*   **CalDAV and CardDAV Support:** Fully compatible with CalDAV and CardDAV standards.
*   **Git-Based Storage:** Store your calendar and contact data in a Git repository.
*   **WebDAV Core Support:** Supports core WebDAV features (excluding LOCK operations).
*   **Docker Integration:**  Ready-to-use Docker image for easy deployment.
*   **Well-Tested Client Compatibility:** Works with a wide range of CalDAV/CardDAV clients.

## Implemented Standards

Xandikos provides support for a wide range of IETF RFC standards, including:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - partial
*   RFC 3744 (Access Control) - partial
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

For detailed compliance information, see the [DAV compliance notes](https://www.xandikos.org/notes/dav-compliance.rst).

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos has been tested and works with a variety of CalDAV/CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   sogo connector for Icedove/Thunderbird
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
*   homeassistant's CalDAV integration
*   pimsync
*   davcli
*   Thunderbird

## Dependencies

Xandikos is built using Python 3 (see `pyproject.toml` for the specific version) and also supports PyPy 3.  It depends on the following Python libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

A Dockerfile is provided for easy deployment. Pre-built images are available on [ghcr.io/jelmer/xandikos](https://github.com/jelmer/xandikos/pkgs/container/xandikos). Configure your Xandikos instance using environment variables, including:

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

See the `examples/docker-compose.yml` file and the [man page](https://www.xandikos.org/manpage.html) for detailed configuration instructions.

## Running

Xandikos can be run directly with a plain HTTP socket or behind a reverse proxy such as Apache or Nginx.  See the `examples/` directory for example init system configurations.

### Testing

To run a standalone instance with default calendar and addressbook in `$HOME/dav`:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

The server will listen on `localhost:8080`.

### Production

Deploy Xandikos in production behind a reverse proxy such as Apache or Nginx.

## Client Instructions

Clients that support RFC:5397 can automatically discover calendar and addressbook URLs.  For clients that require explicit URLs, use the following format:

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are highly encouraged!  Report bugs and suggest features on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Review the [CONTRIBUTING](CONTRIBUTING.md) guide for code and documentation contributions.  Look for the `new-contributor` tag on issues for beginner-friendly tasks.

## Help

Get help and connect with other users via:

*   *#xandikos* IRC channel on the OFTC IRC network.
*   [Xandikos Mailing List](https://groups.google.com/forum/#!forum/xandikos)
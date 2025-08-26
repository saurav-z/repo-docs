# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a self-hosted CalDAV and CardDAV server that stores your calendar and contact data in a Git repository, offering a unique and reliable solution.**

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

[View the original repository on GitHub](https://github.com/jelmer/xandikos)

Xandikos (Ξανδικός or Ξανθικός) takes its name from the March month in the ancient Macedonian calendar.  Find comprehensive documentation on the [official website](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Based Storage:** Leverages the power of Git for versioning, data integrity, and easy backups.
*   **CalDAV & CardDAV Support:** Fully implements core standards for calendar and contact synchronization.
*   **Lightweight & Efficient:** Designed for performance with minimal resource usage.
*   **Docker Support:** Easily deployable with a pre-built Docker image available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos).
*   **Flexible Configuration:** Customizable through command-line arguments, environment variables, and reverse proxy integration.
*   **Extensive Client Compatibility:** Works with a wide range of popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos implements the following RFCs, ensuring broad compatibility:

*   RFC 4918/2518 (Core WebDAV) - *Implemented, except for LOCK operations (COPY/MOVE implemented for non-collections)*
*   RFC 4791 (CalDAV) - *Fully implemented*
*   RFC 6352 (CardDAV) - *Fully implemented*
*   RFC 5397 (Current Principal) - *Fully implemented*
*   RFC 3253 (Versioning Extensions) - *Partially implemented, only the REPORT method and {DAV:}expand-property property*
*   RFC 3744 (Access Control) - *Partially implemented*
*   RFC 5995 (POST to create members) - *Fully implemented*
*   RFC 5689 (Extended MKCOL) - *Fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully implemented*
*   RFC 7953 (Calendar Availability) - *Fully implemented*

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos works with the following CalDAV/CardDAV clients (and many more):

*   Vdirsyncer
*   caldavzap/carddavmate
*   evolution
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

## Dependencies

Xandikos requires Python 3 (see `pyproject.toml` for specific version) and uses the following Python libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker Deployment

A Dockerfile is provided. The Docker image is available at `ghcr.io/jelmer/xandikos`. Release-specific tags (e.g., `v0.2.11`) are also available.

Configure the Docker image using environment variables:

*   `PORT` (default: 8000)
*   `METRICS_PORT` (default: 8001)
*   `LISTEN_ADDRESS` (default: 0.0.0.0)
*   `DATA_DIR` (default: /data)
*   `CURRENT_USER_PRINCIPAL` (default: /user/)
*   `ROUTE_PREFIX` (default: /)
*   `AUTOCREATE` (true/false)
*   `DEFAULTS` (Create default calendar/addressbook - true/false)
*   `DEBUG` (true/false)
*   `DUMP_DAV_XML` (true/false)
*   `NO_STRICT` (true/false)

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more details.

## Running

Xandikos can run directly via HTTP or behind a reverse proxy (Apache, nginx, etc.).

### Testing

To run a standalone instance:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access at `http://localhost:8080`.

Note: Collections require `--defaults` or manual creation via client or Git repository.

### Production

Use a reverse HTTP proxy for production deployments.

See `examples/` for init system configurations.

## Client Configuration

Some clients support automatic discovery using RFC:5397; if not, the URLs are like these if you have used `--defaults`:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Report bugs and request features on [GitHub Issues](https://github.com/jelmer/xandikos/issues/new). See `CONTRIBUTING.md` for details on code contributions. Look for the `new-contributor` tag for beginner-friendly issues.

## Help

*   Join the `#xandikos` IRC channel on OFTC.
*   Join the [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos).
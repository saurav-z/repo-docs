# Xandikos: A Lightweight CardDAV/CalDAV Server Backed by Git

**Xandikos is a powerful and efficient CardDAV/CalDAV server that leverages the versioning capabilities of Git for robust data management.** ([View on GitHub](https://github.com/jelmer/xandikos))

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

## Key Features

*   **Git-Backed Storage:** All data is stored and versioned in a Git repository, providing data integrity, backup, and recovery capabilities.
*   **Full CalDAV and CardDAV Support:** Offers comprehensive support for both CalDAV (calendars) and CardDAV (contacts) protocols, ensuring compatibility with various clients.
*   **Lightweight and Efficient:** Designed for minimal resource consumption, making it suitable for both small and large deployments.
*   **Standards Compliant:** Implements core WebDAV and numerous RFCs for seamless interoperability.
*   **Docker Support:** Easy deployment with a pre-built Docker image, including configuration via environment variables.

## Implemented Standards

Xandikos implements the following standards:

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

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos has been tested and works with a wide range of CalDAV/CardDAV clients:

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

Xandikos is built on Python 3 and depends on:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker Deployment

Xandikos provides a Dockerfile for easy deployment. The image is available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos) and can be configured via environment variables:

*   `PORT`, `METRICS_PORT`, `LISTEN_ADDRESS`, `DATA_DIR`, `CURRENT_USER_PRINCIPAL`, `ROUTE_PREFIX`, `AUTOCREATE`, `DEFAULTS`, `DEBUG`, `DUMP_DAV_XML`, `NO_STRICT`

See the `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more information.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.

### Standalone Example

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a standalone instance on `http://localhost:8080`.

### Production

For production, it's recommended to use a reverse proxy like Apache or Nginx.

## Client Configuration

*   **Auto-discovery:** Some clients support automatic discovery using the base URL.
*   **Manual Configuration:** For clients without auto-discovery, provide the full URL:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new) and refer to the `CONTRIBUTING.md` file for details.
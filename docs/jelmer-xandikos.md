# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a user-friendly CalDAV and CardDAV server that leverages the power of Git for data storage and version control.** ([View on GitHub](https://github.com/jelmer/xandikos))

![Xandikos Logo](logo.png)

## Key Features

*   **Git-Backed Storage:** Utilizes Git for storing and managing calendar and contact data, enabling versioning, backups, and easy data management.
*   **Full CalDAV & CardDAV Support:** Implements core RFC standards for seamless compatibility with a wide range of clients.
*   **Lightweight and Efficient:** Designed for simplicity and performance, making it ideal for personal use or small deployments.
*   **Easy to Deploy:** Supports direct HTTP access, reverse proxy setups, and Docker for flexible deployment options.
*   **Client Compatibility:** Tested and proven to work with popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos implements a wide range of WebDAV, CalDAV, and CardDAV standards, ensuring broad compatibility:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) (Partial)
*   RFC 3744 (Access Control) (Partial)
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos is compatible with a variety of CalDAV and CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
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

Xandikos is built using Python 3 and depends on:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

Xandikos provides a Dockerfile for easy deployment. The Docker image is regularly built and published at `ghcr.io/jelmer/xandikos`. Docker image can be configured via environment variables.

See the [Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos) for more information.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.

### Testing

To run a standalone instance with pre-created data:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access the server at `http://localhost:8080`.

### Production

For production, a reverse proxy like Apache or nginx is recommended. See the `examples/` directory for sample configurations.

## Client Instructions

Clients can automatically discover calendars and address books if they support RFC 5397.  For clients without auto-discovery, use the following URLs (adjusting the domain as needed):

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome!  Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new) and consult the `CONTRIBUTING.md` file.  Look for the `new-contributor` label for good first issues.

## Help

*   IRC: #xandikos on OFTC
*   Mailing List: [Xandikos](https://groups.google.com/forum/#!forum/xandikos)
# Xandikos: Self-Hosted CalDAV/CardDAV Server Backed by Git

**Xandikos is a lightweight, Git-backed server, perfect for syncing your calendars and contacts.**

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

[View the original repository on GitHub](https://github.com/jelmer/xandikos)

Xandikos (Ξανδικός) is named after the Macedonian month of March.

**Key Features:**

*   **Lightweight & Efficient:** Designed for performance and ease of use.
*   **Git-Backed Storage:** Leverages the power and versioning of Git for data storage.
*   **Standards-Compliant:** Implements core CalDAV and CardDAV standards.
*   **Docker Support:** Easily deploy and manage Xandikos using Docker.
*   **Open Source:** Fully open-source and actively maintained.

## Implemented Standards

Xandikos supports the following standards:

*   RFC 4918/RFC 2518 (Core WebDAV) - *Implemented (except for LOCK operations)*
*   RFC 4791 (CalDAV) - *Fully implemented*
*   RFC 6352 (CardDAV) - *Fully implemented*
*   RFC 5397 (Current Principal) - *Fully implemented*
*   RFC 3253 (Versioning Extensions) - *Partially implemented*
*   RFC 3744 (Access Control) - *Partially implemented*
*   RFC 5995 (POST to create members) - *Fully implemented*
*   RFC 5689 (Extended MKCOL) - *Fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully implemented*
*   RFC 7953 (Calendar Availability) - *Fully implemented*

**Not Implemented:**

*   RFC 6638 (CalDAV Scheduling Extensions) - *Not implemented*
*   RFC 7809 (CalDAV Time Zone Extensions) - *Not implemented*
*   RFC 7529 (WebDAV Quota) - *Not implemented*
*   RFC 4709 (WebDAV Mount) - *Intentionally not implemented*
*   RFC 5546 (iCal iTIP) - *Not implemented*
*   RFC 4324 (iCAL CAP) - *Not implemented*

For detailed compliance information, see the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos has been tested and is compatible with a wide range of CalDAV/CardDAV clients, including:

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

## Dependencies

Xandikos is built using Python 3 and requires the following dependencies:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install these dependencies using `pip` or your system's package manager (e.g., `apt` on Debian/Ubuntu).

## Docker

Xandikos provides a Docker image for easy deployment. The image is regularly built and published at `ghcr.io/jelmer/xandikos`.  Environment variables are available for configuration, including:

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

See the Dockerfile comments, `examples/docker-compose.yml`, and the [man page](https://www.xandikos.org/manpage.html) for detailed configuration instructions.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy (like Apache or nginx).

**Standalone Example:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a standalone instance listening on `localhost:8080`.

## Client Instructions

Clients that support RFC:`5397` can automatically discover calendar and address book URLs. For other clients, use these example URLs, adjusting the domain as needed:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Report bugs and request features on [GitHub](https://github.com/jelmer/xandikos/issues/new). See `CONTRIBUTING.md` for guidelines.  Look for the `new-contributor` tag for beginner-friendly issues.

## Help

Get help via the `#xandikos` IRC channel on the OFTC IRC network or the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
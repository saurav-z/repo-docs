# Xandikos: Your Lightweight, Git-Backed CardDAV/CalDAV Server

Xandikos is a powerful and easy-to-use CardDAV/CalDAV server that stores your data in a Git repository, offering robust versioning and data management. **[Learn more and contribute on GitHub](https://github.com/jelmer/xandikos)**.

![Xandikos Logo](logo.png)

## Key Features

*   **Lightweight and Efficient:** Designed for performance with minimal resource usage.
*   **Git-Backed Storage:** Leverages Git for version control, backups, and data integrity.
*   **Comprehensive Standard Compliance:** Implements a wide range of WebDAV, CalDAV, and CardDAV standards.
*   **Flexible Deployment:** Supports direct HTTP listening and reverse proxy setups.
*   **Docker Support:** Easily deploy and configure Xandikos using Docker.
*   **Easy Configuration:** Configure using environment variables.
*   **Broad Client Compatibility:** Works seamlessly with popular CalDAV/CardDAV clients.

## Implemented Standards

*   RFC 4918/2518 (Core WebDAV) (Implemented, except for LOCK operations)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) (Partially)
*   RFC 3744 (Access Control) (Partially)
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

## Not Implemented Standards

*   RFC 6638 (CalDAV Scheduling Extensions)
*   RFC 7809 (CalDAV Time Zone Extensions)
*   RFC 7529 (WebDAV Quota)
*   RFC 4709 (WebDAV Mount) (intentionally not implemented)
*   RFC 5546 (iCal iTIP)
*   RFC 4324 (iCAL CAP)

For detailed compliance information, see the [DAV compliance notes](https://www.xandikos.org/docs/)

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos has been tested with the following clients:

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

*   Python 3 (specified in pyproject.toml)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

A Dockerfile is provided for easy deployment. The image is available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos) and can be configured using environment variables.

## Running

Xandikos can be run directly or behind a reverse proxy.  See the examples in the repo.

## Client Instructions

Clients supporting RFC:5397 can use the base URL for setup.  For clients requiring direct URLs, use the following structure:

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome!  Please file issues on [GitHub](https://github.com/jelmer/xandikos/issues/new) and read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.
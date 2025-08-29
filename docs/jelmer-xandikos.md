# Xandikos: Your Lightweight Git-Backed CalDAV/CardDAV Server

**Xandikos is a self-hosted, lightweight CalDAV/CardDAV server that leverages the power of Git for data storage and versioning.**  [View on GitHub](https://github.com/jelmer/xandikos)

![Xandikos Logo](logo.png)

Xandikos (Ξανδικός or Ξανθικός) takes its name from the March month in the ancient Macedonian calendar. It provides a robust and efficient solution for managing your calendar and contact data, backed by the reliability of a Git repository. Detailed documentation can be found on the [Xandikos Home Page](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Backed Storage:**  Uses Git for versioning and data storage, ensuring data integrity and providing easy backups.
*   **Complete Standards Compliance:** Implements core WebDAV, CalDAV, and CardDAV standards.
*   **Lightweight & Efficient:** Designed for performance and minimal resource usage.
*   **Docker Support:** Easy to deploy and manage with Docker containers.
*   **Flexible Deployment:** Can run directly on HTTP or behind a reverse proxy.

## Implemented Standards

Xandikos supports a wide range of standards:

*   RFC 4918/2518 (Core WebDAV) - (excluding LOCK)
*   RFC 4791 (CalDAV) - Fully Implemented
*   RFC 6352 (CardDAV) - Fully Implemented
*   RFC 5397 (Current Principal) - Fully Implemented
*   RFC 3253 (Versioning Extensions) - Partially Implemented
*   RFC 3744 (Access Control) - Partially Implemented
*   RFC 5995 (POST to create members) - Fully Implemented
*   RFC 5689 (Extended MKCOL) - Fully Implemented
*   RFC 6578 (Collection Synchronization for WebDAV) - Fully Implemented
*   RFC 7953 (Calendar Availability) - Fully Implemented

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos works seamlessly with a variety of CalDAV and CardDAV clients, including:

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

Xandikos is built using Python 3 (see pyproject.toml for version) and uses the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

A Dockerfile is provided, and images are regularly built and published at `ghcr.io/jelmer/xandikos`.

*   **Image Tags:** `v$RELEASE` tags are available for specific releases (e.g., `v0.2.11`).
*   **Configuration:** Configurable via environment variables (PORT, METRICS_PORT, LISTEN_ADDRESS, DATA_DIR, etc.).
*   **Examples:** See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more information.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.

**Testing:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a standalone instance on `localhost:8080`.  You can create collections from your CalDAV/CardDAV client, or by creating git repositories under the *contacts* or *calendars* directories.

**Production:**

It is recommended to run Xandikos behind a reverse HTTP proxy like Apache or nginx.

## Client Instructions

Some clients can automatically discover the calendar and address book URLs. For those that don't, the URLs will be something like this:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome!  Please report bugs or request features on [GitHub](https://github.com/jelmer/xandikos/issues/new) and read the [CONTRIBUTING.md](CONTRIBUTING.md) file. Issues tagged as `new-contributor` are good starting points.

## Get Help

*   **IRC:**  Join the *#xandikos* channel on the [OFTC](https://www.oftc.net/) IRC network.
*   **Mailing List:**  [Xandikos Mailing List](https://groups.google.com/forum/#!forum/xandikos)
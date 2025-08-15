# Xandikos: A Lightweight, Git-Backed CalDAV/CardDAV Server

**Xandikos is a powerful and easy-to-use CalDAV/CardDAV server that stores your calendar and contact data in a Git repository.** ([Original Repository](https://github.com/jelmer/xandikos))

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

Get started with Xandikos to manage your calendar and contact data securely and efficiently.

**Key Features:**

*   **Git-Backed:** Stores your data in a Git repository for version control, backups, and easy data migration.
*   **Standards Compliant:** Implements a wide range of CalDAV and CardDAV standards (see below).
*   **Lightweight:** Designed for simplicity and efficiency, making it easy to set up and manage.
*   **Docker Support:** Easily deploy Xandikos using Docker containers.
*   **Client Compatibility:** Works with a variety of popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos supports a comprehensive set of CalDAV and CardDAV standards:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *implemented, except for LOCK operations (COPY/MOVE implemented for non-collections)*
*   :RFC:`4791` (CalDAV) - *fully implemented*
*   :RFC:`6352` (CardDAV) - *fully implemented*
*   :RFC:`5397` (Current Principal) - *fully implemented*
*   :RFC:`3253` (Versioning Extensions) - *partially implemented, only the REPORT method and {DAV:}expand-property property*
*   :RFC:`3744` (Access Control) - *partially implemented*
*   :RFC:`5995` (POST to create members) - *fully implemented*
*   :RFC:`5689` (Extended MKCOL) - *fully implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *fully implemented*
*   :RFC:`7953` (Calendar Availability) - *fully implemented*

The following standards are *not* implemented:

*   :RFC:`6638` (CalDAV Scheduling Extensions) - *not implemented*
*   :RFC:`7809` (CalDAV Time Zone Extensions) - *not implemented*
*   :RFC:`7529` (WebDAV Quota) - *not implemented*
*   :RFC:`4709` (WebDAV Mount) - `intentionally <https://github.com/jelmer/xandikos/issues/48>`_ *not implemented*
*   :RFC:`5546` (iCal iTIP) - *not implemented*
*   :RFC:`4324` (iCAL CAP) - *not implemented*

For more detailed information on specification compliance, see the `DAV compliance <notes/dav-compliance.rst>`_ notes.

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos works with a wide range of CalDAV/CardDAV clients, including:

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

Xandikos requires Python 3 (specific version in pyproject.toml) and PyPy 3, along with these dependencies:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Docker

Xandikos provides a Dockerfile for easy deployment. The image is regularly built and published at `ghcr.io/jelmer/xandikos`. Use the `v$RELEASE` tags for specific releases (e.g., `v0.2.11`).

**Configuration:**

The Docker image can be configured using environment variables, including:

*   `PORT` (default: 8000)
*   `METRICS_PORT` (default: 8001)
*   `LISTEN_ADDRESS` (default: 0.0.0.0)
*   `DATA_DIR` (default: /data)
*   `CURRENT_USER_PRINCIPAL` (default: /user/)
*   `ROUTE_PREFIX` (default: /)
*   `AUTOCREATE` (true/false)
*   `DEFAULTS` (Create default calendar/addressbook - true/false)
*   `DEBUG` (Enable debug logging - true/false)
*   `DUMP_DAV_XML` (Print DAV XML requests/responses - true/false)
*   `NO_STRICT` (Enable client compatibility workarounds - true/false)

See `examples/docker-compose.yml` and the `man page <https://www.xandikos.org/manpage.html>`_ for more details.

## Running

Xandikos can run directly or behind a reverse proxy.

**Standalone Example:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This creates a standalone instance with a pre-created calendar and address book, storing data in `$HOME/dav` and listening on `localhost:8080`.

Xandikos creates collections only when `--defaults` is specified or when you create them from your CalDAV/CardDAV client. Alternatively, create git repositories in the `contacts` or `calendars` directories.

**Production:**

For production, use a reverse HTTP proxy like Apache or nginx. See the `examples/` directory for init system configurations.

## Client Instructions

Some clients support automatic calendar/address book discovery. For these, provide the base Xandikos URL. Clients without discovery need the direct URL:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Please submit issues and feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new). Review `CONTRIBUTING.md` for code and documentation contributions. Issues suitable for new contributors are tagged `new-contributor <https://github.com/jelmer/xandikos/labels/new-contributor>`_.

## Help

Join the community:

*   IRC: `#xandikos` on OFTC
*   Mailing List: [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
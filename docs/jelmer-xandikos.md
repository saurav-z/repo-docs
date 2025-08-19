# Xandikos: Your Lightweight, Git-Backed CalDAV/CardDAV Server

**Xandikos is a powerful and flexible CalDAV and CardDAV server that leverages the versioning capabilities of Git for data storage.** ([Original Repository](https://github.com/jelmer/xandikos))

![Xandikos Logo](logo.png)

## Key Features

*   **Standards Compliant:** Fully implements core WebDAV, CalDAV, and CardDAV standards for robust compatibility.
*   **Git-Backed Storage:** Utilizes Git repositories for efficient and reliable data storage, enabling easy backups and version control.
*   **Lightweight and Efficient:** Designed to be lightweight and performant, making it suitable for various hosting environments.
*   **Docker Support:** Includes a Dockerfile for easy deployment and containerization.
*   **Extensive Client Compatibility:** Works seamlessly with a wide range of CalDAV and CardDAV clients, including Vdirsyncer, DAVx5, and many more.

## Implemented Standards

Xandikos supports a comprehensive set of WebDAV, CalDAV, and CardDAV standards:

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

## Not Implemented Standards

*   RFC 6638 (CalDAV Scheduling Extensions)
*   RFC 7809 (CalDAV Time Zone Extensions)
*   RFC 7529 (WebDAV Quota)
*   RFC 4709 (WebDAV Mount) - *Intentionally not implemented*
*   RFC 5546 (iCal iTIP)
*   RFC 4324 (iCAL CAP)

For more details on specification compliance, see the `DAV compliance notes <notes/dav-compliance.rst>`_.

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos has been tested and works with the following CalDAV/CardDAV clients:

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

Xandikos is built with the following Python dependencies:

*   Python 3 (see pyproject.toml for specific version)
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip with: `python setup.py develop`.

## Docker

A Dockerfile is provided for easy deployment. The image is available at `ghcr.io/jelmer/xandikos`. For more information see the `Container overview page <https://github.com/jelmer/xandikos/pkgs/container/xandikos>`.

### Configuration

The Docker image can be configured using environment variables, including:

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

See `examples/docker-compose.yml` and the `man page <https://www.xandikos.org/manpage.html>`_ for more information.

## Running

Xandikos can be run directly or behind a reverse proxy.

### Standalone Example

To run a standalone instance with a pre-created calendar and address book:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create data at `$HOME/dav` and serve on `localhost:8080`.

### Production

For production, it's recommended to use a reverse proxy like Apache or Nginx in front of Xandikos. See the `examples/` directory for init system configurations.

## Client Instructions

Clients can automatically discover calendars and address books from Xandikos if they support RFC:`5397`. For clients requiring direct URLs, the base URL should be provided, e.g., `http://dav.example.com/user/calendars/calendar` and `http://dav.example.com/user/contacts/addressbook`.

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub. See `CONTRIBUTING.md` for details.
Issues suitable for new contributors are tagged `new-contributor <https://github.com/jelmer/xandikos/labels/new-contributor>`_.

## Help

*   IRC: *#xandikos* on the `OFTC <https://www.oftc.net/>`_ network
*   Mailing List: `Xandikos <https://groups.google.com/forum/#!forum/xandikos>`_
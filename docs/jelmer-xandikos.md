# Xandikos: Lightweight CardDAV/CalDAV Server with Git Backend

**Xandikos is a powerful yet simple CalDAV/CardDAV server that uses a Git repository for data storage, offering a reliable and flexible solution for managing your calendars and contacts.**  [View on GitHub](https://github.com/jelmer/xandikos)

![Xandikos Logo](logo.png)

Explore comprehensive documentation on the [Xandikos home page](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Based Storage:** Utilizes Git for version control and data persistence, ensuring data integrity and easy backups.
*   **Standards Compliance:** Implements a wide range of CalDAV/CardDAV standards.
*   **Client Compatibility:** Works seamlessly with popular CalDAV/CardDAV clients.
*   **Docker Support:** Easily deployable with a provided Dockerfile and pre-built images on Docker Hub.
*   **Flexible Configuration:** Configurable via command-line arguments, environment variables, and reverse proxy support.

## Implemented Standards

Xandikos supports the following standards:

*   RFC 4918/2518 (Core WebDAV) (Implemented, except for LOCK operations)
*   RFC 4791 (CalDAV) - *fully implemented*
*   RFC 6352 (CardDAV) - *fully implemented*
*   RFC 5397 (Current Principal) - *fully implemented*
*   RFC 3253 (Versioning Extensions) - *partially implemented*
*   RFC 3744 (Access Control) - *partially implemented*
*   RFC 5995 (POST to create members) - *fully implemented*
*   RFC 5689 (Extended MKCOL) - *fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *fully implemented*
*   RFC 7953 (Calendar Availability) - *fully implemented*

See [DAV compliance](notes/dav-compliance.rst) for detailed compliance information.

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos is compatible with a variety of CalDAV/CardDAV clients, including:

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

Xandikos requires Python 3 and uses the following dependencies:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

For Debian, install dependencies using: `sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2`
Or install using pip: `python setup.py develop`

## Docker

Xandikos provides a Dockerfile for easy deployment. Published images are available on GitHub Container Registry `ghcr.io/jelmer/xandikos`. Use tags like `v0.2.11` for specific releases.

Configure the Docker image using environment variables:

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

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for details.

## Running

Xandikos can be run directly or behind a reverse HTTP proxy.

**Testing:**

Run a standalone instance with a pre-created calendar and addressbook (storing data in *$HOME/dav*):

```bash
./bin/xandikos --defaults -d $HOME/dav
```
Access your server at http://localhost:8080/

**Production:**

Use a reverse HTTP proxy like Apache or nginx.  See examples/ for init system configurations.

## Client Instructions

Some clients can automatically discover the calendar and addressbook URLs from a DAV server (if they support RFC:5397). For such clients you can simply provide the base URL to Xandikos during setup.

For clients that lack automated discovery, use these URLs (assuming you used `--defaults`):

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome!  Report issues and suggest features on [GitHub Issues](https://github.com/jelmer/xandikos/issues/new).  Read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Look for issues tagged `new-contributor` for entry-level tasks.

## Help

*   [OFTC IRC](https://www.oftc.net/) IRC channel: #xandikos
*   [Xandikos Mailing List](https://groups.google.com/forum/#!forum/xandikos)
<!-- SEO-optimized README for Xandikos -->

# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a powerful yet simple CalDAV and CardDAV server that stores your calendar and contact data in a Git repository, offering version control and easy backups.** 

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

[View the original repository on GitHub](https://github.com/jelmer/xandikos)

## Key Features

*   **Git-Backed Storage:** Leverages the power of Git for versioning, backups, and data integrity.
*   **Complete Standard Compliance:** Implements core WebDAV, CalDAV, and CardDAV standards for broad client compatibility.
*   **Lightweight and Efficient:** Designed for ease of use and minimal resource consumption.
*   **Docker Support:** Easily deployable using Docker for simplified setup and management.
*   **Extensive Client Compatibility:** Works seamlessly with a wide range of CalDAV and CardDAV clients.
*   **Flexible Configuration:** Supports various deployment scenarios, including direct HTTP and reverse proxy setups.

## Implemented Standards

Xandikos supports a wide range of standards, ensuring compatibility with various clients:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - Partial Implementation
*   RFC 3744 (Access Control) - Partial Implementation
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

For more details on specification compliance, see the [DAV compliance notes](notes/dav-compliance.rst).

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos is compatible with numerous CalDAV and CardDAV clients, including:

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
*   Home Assistant CalDAV integration
*   pimsync
*   davcli
*   Thunderbird
*   ...and many more!

## Getting Started

### Dependencies

Xandikos requires Python 3 (check pyproject.toml for specific version) and supports PyPy 3. It also uses the following Python libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install these dependencies using pip:

```bash
python setup.py develop
```

or for Debian-based systems:
```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

### Docker

Xandikos provides a Dockerfile for easy deployment.  Pre-built images are available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

**Docker Configuration:**

Configure the Docker image using environment variables:

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

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more details.

### Running

Xandikos can be run directly or behind a reverse proxy.

**Standalone (for testing):**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access the server at `http://localhost:8080`.  Note that `--defaults` creates default calendar and addressbook directories.

**Production:**

Use a reverse HTTP proxy like Apache or nginx in front of Xandikos. See examples/ for init system configurations.

## Client Configuration

Some clients automatically discover calendar and addressbook URLs (RFC 5397 support). Others require the direct URL. If you used `--defaults`, your URLs will be:

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome!  Report bugs and request features on [GitHub](https://github.com/jelmer/xandikos/issues/new). Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  Look for issues tagged `new-contributor` to get started.

## Support

*   [OFTC IRC channel](https://www.oftc.net/) - #xandikos
*   [Xandikos Mailing List](https://groups.google.com/forum/#!forum/xandikos)

## Documentation

Find extended documentation and usage information [here](https://www.xandikos.org/docs/).
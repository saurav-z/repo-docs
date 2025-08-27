# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a powerful and lightweight server, seamlessly merging CalDAV and CardDAV functionalities with the versioning capabilities of a Git repository.**

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

[Explore the Xandikos Repository](https://github.com/jelmer/xandikos)

## Key Features

*   **Git-backed Storage:** Utilizes a Git repository for robust, versioned storage of your calendar and contact data.
*   **Standards Compliant:** Implements core WebDAV, CalDAV, and CardDAV standards, ensuring broad compatibility.
*   **Lightweight:** Designed for efficiency, making it suitable for resource-constrained environments.
*   **Easy Deployment:** Offers multiple deployment options, including direct HTTP and Docker.
*   **Broad Client Compatibility:** Works with popular CalDAV/CardDAV clients (see list below).

## Implemented Standards

Xandikos provides comprehensive support for a wide range of standards:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *Implemented (excluding LOCK operations)*
*   :RFC:`4791` (CalDAV) - *Fully implemented*
*   :RFC:`6352` (CardDAV) - *Fully implemented*
*   :RFC:`5397` (Current Principal) - *Fully implemented*
*   :RFC:`3253` (Versioning Extensions) - *Partially implemented*
*   :RFC:`3744` (Access Control) - *Partially implemented*
*   :RFC:`5995` (POST to create members) - *Fully implemented*
*   :RFC:`5689` (Extended MKCOL) - *Fully implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *Fully implemented*
*   :RFC:`7953` (Calendar Availability) - *Fully implemented*

## Not Implemented Standards

*   :RFC:`6638` (CalDAV Scheduling Extensions)
*   :RFC:`7809` (CalDAV Time Zone Extensions)
*   :RFC:`7529` (WebDAV Quota)
*   :RFC:`4709` (WebDAV Mount) - *Intentionally not implemented*
*   :RFC:`5546` (iCal iTIP)
*   :RFC:`4324` (iCAL CAP)

For more details on specification compliance, see the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos is compatible with a wide range of CalDAV and CardDAV clients:

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

Xandikos relies on the following Python libraries:

*   Python 3 (see `pyproject.toml` for specific version)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

### Installation example for Debian

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

or with pip:

```bash
python setup.py develop
```

## Docker

Xandikos offers a Docker image for easy deployment:

*   Available on ghcr.io/jelmer/xandikos.
*   Tagged with releases, e.g., v0.2.11 for release 0.2.11.

See the [Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos) for a full list.

### Docker Configuration

Configure the Docker image using environment variables:

*   ``PORT`` - Listening port (default: 8000)
*   ``METRICS_PORT`` - Metrics endpoint (default: 8001)
*   ``LISTEN_ADDRESS`` - Bind address (default: 0.0.0.0)
*   ``DATA_DIR`` - Data directory (default: /data)
*   ``CURRENT_USER_PRINCIPAL`` - User principal path (default: /user/)
*   ``ROUTE_PREFIX`` - URL route prefix (default: /)
*   ``AUTOCREATE`` - Auto-create directories (true/false)
*   ``DEFAULTS`` - Create default calendar/addressbook (true/false)
*   ``DEBUG`` - Enable debug logging (true/false)
*   ``DUMP_DAV_XML`` - Print DAV XML requests/responses (true/false)
*   ``NO_STRICT`` - Enable client compatibility workarounds (true/false)

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more details.

## Running Xandikos

Xandikos can run directly or behind a reverse proxy.

### Standalone Testing

Run a standalone instance with a pre-created calendar and addressbook:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access the server at `http://localhost:8080`.  No collections are created unless `--defaults` is used.  You can also create collections from your CalDAV/CardDAV client, or by creating git repositories under the *contacts* or *calendars* directories.

### Production Deployment

Recommended to run behind a reverse HTTP proxy like Apache or Nginx. See `examples/` for init system configurations.

## Client Configuration

Clients can automatically discover calendar and addressbook URLs if they support RFC:`5397`. If not, use these URLs (if you used `--defaults`):

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome! Please submit issues and feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new). Review the [CONTRIBUTING](CONTRIBUTING.md) guide for code and documentation contributions. Issues suitable for new contributors are tagged `new-contributor`.

## Help

*   Join the *#xandikos* IRC channel on the OFTC IRC network.
*   Use the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
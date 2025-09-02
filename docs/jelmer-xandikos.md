<!--  Title: Xandikos - Lightweight, Git-Backed CalDAV/CardDAV Server -->

# Xandikos: Your Self-Hosted Calendar and Contact Server

**Xandikos is a lightweight, yet powerful, CalDAV/CardDAV server that uses a Git repository for data storage, making it easy to back up, version, and manage your calendar and contact information.** [Explore the original repository on GitHub](https://github.com/jelmer/xandikos).

![Xandikos Logo](logo.png)

_Find more detailed information and documentation on the [official Xandikos website](https://www.xandikos.org/docs/)._

## Key Features

*   **Git-Based Storage:** Leverages Git for versioning, backups, and easy data management.
*   **CalDAV & CardDAV Support:** Fully compliant with RFC standards for calendar and contact synchronization.
*   **Lightweight & Efficient:** Designed for simplicity and speed.
*   **Docker Support:** Easily deploy and manage Xandikos using Docker containers.
*   **Flexible Configuration:** Customizable via command-line arguments and environment variables.
*   **Open Source:**  Freely available under an open-source license.

## Implemented Standards

Xandikos supports a wide range of WebDAV, CalDAV, and CardDAV standards:

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

**Not Implemented:**

*   :RFC:`6638` (CalDAV Scheduling Extensions)
*   :RFC:`7809` (CalDAV Time Zone Extensions)
*   :RFC:`7529` (WebDAV Quota)
*   :RFC:`4709` (WebDAV Mount) - *Intentionally not implemented*
*   :RFC:`5546` (iCal iTIP)
*   :RFC:`4324` (iCAL CAP)

See `DAV compliance <notes/dav-compliance.rst>` for more detailed compliance information.

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos is compatible with a wide range of CalDAV and CardDAV clients:

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
*   Home Assistant CalDAV integration
*   pimsync
*   davcli
*   Thunderbird

## Dependencies

Xandikos is written in Python 3 and utilizes the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

**Installation (example with pip):**
```bash
python setup.py develop
```

## Docker Deployment

Xandikos provides a Dockerfile for easy deployment. Pre-built images are available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

**Configuration:**

Configure the Docker image using environment variables:

*   ``PORT`` (default: 8000)
*   ``METRICS_PORT`` (default: 8001)
*   ``LISTEN_ADDRESS`` (default: 0.0.0.0)
*   ``DATA_DIR`` (default: /data)
*   ``CURRENT_USER_PRINCIPAL`` (default: /user/)
*   ``ROUTE_PREFIX`` (default: /)
*   ``AUTOCREATE`` (true/false)
*   ``DEFAULTS`` (Create default calendar/addressbook)
*   ``DEBUG`` (Enable debug logging)
*   ``DUMP_DAV_XML`` (Print DAV XML requests/responses)
*   ``NO_STRICT`` (Enable client compatibility workarounds)

Refer to `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for more details.

## Running Xandikos

Xandikos can run directly with HTTP or behind a reverse proxy.

**Testing:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access the server at `http://localhost:8080/`.

**Production:**
Deploy with a reverse HTTP proxy like Apache or nginx. Example init system configurations are available in the `examples/` directory.

## Client Setup

Some clients automatically discover URLs.  For those that don't, use these example URLs:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome!  Please submit issues and feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Read the [CONTRIBUTING](CONTRIBUTING.md) guidelines for code and documentation contributions. Issues tagged as `new-contributor` are great for newcomers.

## Help & Support

*   Join the *#xandikos* IRC channel on the [OFTC](https://www.oftc.net/) IRC network.
*   Subscribe to the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
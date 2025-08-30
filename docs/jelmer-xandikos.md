# Xandikos: A Lightweight Git-Backed CalDAV/CardDAV Server

**Tired of complex CalDAV/CardDAV servers? Xandikos is the simple, powerful, and reliable solution that stores your calendar and contact data in a Git repository.**  [Check it out on GitHub!](https://github.com/jelmer/xandikos)

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

## Key Features

*   **Lightweight & Efficient:** Designed for simplicity and performance.
*   **Git-Backed Storage:** Leverages Git for versioning, backup, and easy data management.
*   **Complete Standard Compliance:** Fully implements core CalDAV and CardDAV RFCs.
*   **Docker Support:** Easily deployable using Docker containers.
*   **Client Compatibility:** Works with a wide range of popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos supports a comprehensive set of WebDAV, CalDAV, and CardDAV standards:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *Implemented (except LOCK)*
*   :RFC:`4791` (CalDAV) - *Fully Implemented*
*   :RFC:`6352` (CardDAV) - *Fully Implemented*
*   :RFC:`5397` (Current Principal) - *Fully Implemented*
*   :RFC:`3253` (Versioning Extensions) - *Partially Implemented*
*   :RFC:`3744` (Access Control) - *Partially Implemented*
*   :RFC:`5995` (POST to create members) - *Fully Implemented*
*   :RFC:`5689` (Extended MKCOL) - *Fully Implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *Fully Implemented*
*   :RFC:`7953` (Calendar Availability) - *Fully Implemented*

For more detailed information about RFC compliance, see `DAV compliance <notes/dav-compliance.rst>`_.

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos has been tested and works with many popular CalDAV/CardDAV clients, including:

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

Xandikos is built on Python 3 and utilizes the following dependencies:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

Xandikos offers a Docker image for easy deployment and configuration. Pre-built images are available at `ghcr.io/jelmer/xandikos`.

The Docker image can be configured with these environment variables:

*   ``PORT`` (default: 8000)
*   ``METRICS_PORT`` (default: 8001)
*   ``LISTEN_ADDRESS`` (default: 0.0.0.0)
*   ``DATA_DIR`` (default: /data)
*   ``CURRENT_USER_PRINCIPAL`` (default: /user/)
*   ``ROUTE_PREFIX`` (default: /)
*   ``AUTOCREATE``
*   ``DEFAULTS``
*   ``DEBUG``
*   ``DUMP_DAV_XML``
*   ``NO_STRICT``

For detailed instructions, see the Dockerfile and `examples/docker-compose.yml`. More info on the image at the [Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

## Running Xandikos

You can run Xandikos directly or behind a reverse proxy.

**Standalone Instance (for testing):**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will start a server listening on `localhost:8080`.

**Production:**
For production, using a reverse HTTP proxy like Apache or nginx is recommended.

For example init system configurations, see examples/.

## Client Configuration

Clients can often auto-discover calendar and addressbook URLs.  If your client requires direct URLs, they will look something like:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome!  Report bugs and request features on [GitHub](https://github.com/jelmer/xandikos/issues/new). Review the [CONTRIBUTING](CONTRIBUTING.md) guidelines for code contributions.

## Help

*   IRC: `#xandikos` on the `OFTC <https://www.oftc.net/>`_ IRC network.
*   Mailing List: `Xandikos <https://groups.google.com/forum/#!forum/xandikos>`_
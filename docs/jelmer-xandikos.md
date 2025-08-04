# Xandikos: A Lightweight Git-Backed CalDAV/CardDAV Server

**Xandikos is a powerful and flexible CalDAV/CardDAV server that stores your calendar and contact data in a Git repository, offering robust versioning and easy data management.**

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)  
*(Logo will display here)*

[View the Xandikos repository on GitHub](https://github.com/jelmer/xandikos)

## Key Features:

*   **Git-Based Storage:** Utilizes Git for versioning, allowing for easy backups, rollbacks, and data management.
*   **Complete CalDAV/CardDAV Support:** Fully implements core standards for calendar and contact synchronization.
*   **Lightweight and Efficient:** Designed for optimal performance and minimal resource usage.
*   **Docker Support:** Easily deployable with Docker for simplified setup and management.
*   **Flexible Deployment:** Can run directly or behind a reverse proxy.
*   **Extensive Client Compatibility:** Works with a wide range of CalDAV/CardDAV clients.

## Implemented Standards:

Xandikos supports a comprehensive set of WebDAV, CalDAV, and CardDAV standards:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *Implemented (except LOCK operations)*
*   :RFC:`4791` (CalDAV) - *Fully Implemented*
*   :RFC:`6352` (CardDAV) - *Fully Implemented*
*   :RFC:`5397` (Current Principal) - *Fully Implemented*
*   :RFC:`3253` (Versioning Extensions) - *Partially Implemented*
*   :RFC:`3744` (Access Control) - *Partially Implemented*
*   :RFC:`5995` (POST to create members) - *Fully Implemented*
*   :RFC:`5689` (Extended MKCOL) - *Fully Implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *Fully Implemented*
*   :RFC:`7953` (Calendar Availability) - *Fully Implemented*

## Not Implemented Standards:

*   :RFC:`6638` (CalDAV Scheduling Extensions)
*   :RFC:`7809` (CalDAV Time Zone Extensions)
*   :RFC:`7529` (WebDAV Quota)
*   :RFC:`4709` (WebDAV Mount) - *Intentionally Not Implemented*
*   :RFC:`5546` (iCal iTIP)
*   :RFC:`4324` (iCAL CAP)

For detailed information on specification compliance, see `DAV compliance <notes/dav-compliance.rst>`_.

## Limitations:

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients:

Xandikos works with a variety of CalDAV/CardDAV clients, including:

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

## Dependencies:

Xandikos is built using Python 3 and depends on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip or your system package manager (e.g., `sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2` on Debian/Ubuntu).

## Docker:

A Dockerfile is provided for easy deployment. The Docker image is regularly built and published at `ghcr.io/jelmer/xandikos`.  Tagged releases are available (e.g., `v0.2.11`). See `the Container overview page <https://github.com/jelmer/xandikos/pkgs/container/xandikos>`_ for a full list.

## Running Xandikos:

You can run Xandikos directly or behind a reverse HTTP proxy.

**Standalone Testing:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```
This starts a server on `localhost:8080`.

**Production:**

Use a reverse HTTP proxy like Apache or nginx in front of Xandikos. See `examples/` for example init system configurations.

## Client Instructions:

Clients that support automatic discovery can use the base URL. For clients requiring specific URLs, use the following format:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing:

Contributions are welcome!  Please submit issues and pull requests on GitHub:  [https://github.com/jelmer/xandikos/issues/new](https://github.com/jelmer/xandikos/issues/new).  See `CONTRIBUTING <CONTRIBUTING.md>`_ for more details.  Look for the `new-contributor <https://github.com/jelmer/xandikos/labels/new-contributor>`_ label on issues to find good starting points.

## Help:

*   Join the *#xandikos* IRC channel on the `OFTC <https://www.oftc.net/>`_ network.
*   Visit the `Xandikos <https://groups.google.com/forum/#!forum/xandikos>`_ mailing list.
*   Explore the `man page <https://www.xandikos.org/manpage.html>`_
*   Read the extended documentation on the home page <https://www.xandikos.org/docs/>`_.
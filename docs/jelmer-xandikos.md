# Xandikos: A Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a powerful, self-hosted CalDAV and CardDAV server that leverages the version control capabilities of Git, offering robust data management and easy backups.**  [View the original repository](https://github.com/jelmer/xandikos)

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

[Explore the extended documentation](https://www.xandikos.org/docs/)

## Key Features

*   **Git-backed storage:** Benefit from Git's versioning and backup capabilities for your calendar and contact data.
*   **Full CalDAV and CardDAV support:** Compatible with a wide range of clients for calendar and contact synchronization.
*   **Lightweight and easy to set up:** Simple deployment, ideal for personal or small-team use.
*   **Docker support:** Deploy easily using pre-built Docker images.
*   **Standards Compliant:** Implements a wide range of RFCs, ensuring compatibility.

## Implemented Standards

Xandikos implements a comprehensive set of WebDAV, CalDAV, and CardDAV standards:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *Implemented (except LOCK operations)*
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

*   :RFC:`6638` (CalDAV Scheduling Extensions) - *Not implemented*
*   :RFC:`7809` (CalDAV Time Zone Extensions) - *Not implemented*
*   :RFC:`7529` (WebDAV Quota) - *Not implemented*
*   :RFC:`4709` (WebDAV Mount) - *Intentionally not implemented*
*   :RFC:`5546` (iCal iTIP) - *Not implemented*
*   :RFC:`4324` (iCAL CAP) - *Not implemented*

For a detailed breakdown of specification compliance, see `DAV compliance <notes/dav-compliance.rst>`_.

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos is compatible with a wide variety of CalDAV and CardDAV clients, including:

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

Xandikos is built with Python 3 (check `pyproject.toml` for specific versions) and PyPy 3.  It relies on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip or apt-get as shown in the original README.

## Docker

Xandikos provides a Dockerfile for easy deployment.  The image is regularly built and published on `ghcr.io/jelmer/xandikos`.  Use tags like `v0.2.11` for specific releases.

The Docker image can be configured with environment variables, see the original README for all available configurations.

## Running Xandikos

Xandikos can run directly or behind a reverse proxy like Apache or Nginx.

**Standalone Testing:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will create a local instance on `localhost:8080`.

**Production:**

For production, using a reverse proxy is recommended. Example init system configurations can be found in the `examples/` directory.

## Client Configuration

Some clients automatically discover calendar and address book URLs (RFC:5397). For those, the base URL of your Xandikos server is sufficient.  For others, you'll need the direct URLs:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome! Please submit issues and feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for code and documentation contributions.  Look for `new-contributor` tagged issues for good starting points.

## Help and Support

*   Join the *#xandikos* IRC channel on the OFTC network.
*   Check out the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
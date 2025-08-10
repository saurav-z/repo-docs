# Xandikos: Your Lightweight Git-Backed CalDAV/CardDAV Server

**Tired of complex calendar and contact servers? Xandikos offers a simple yet powerful solution, leveraging the version control of Git for your CalDAV and CardDAV data.** [View the project on GitHub](https://github.com/jelmer/xandikos)

![Xandikos Logo](logo.png)

**Key Features:**

*   **Git-Backed Data:** Uses Git for data storage, enabling versioning, backups, and easy management.
*   **CalDAV and CardDAV Support:** Fully implements CalDAV and CardDAV standards for calendar and contact synchronization.
*   **Lightweight and Efficient:** Designed for simplicity and performance, making it ideal for personal use or small teams.
*   **Docker Support:** Easily deployable using Docker, simplifying setup and management.
*   **Wide Client Compatibility:** Works with numerous CalDAV/CardDAV clients, ensuring seamless integration with your existing devices and applications.
*   **Flexible Deployment:** Can be run directly or behind a reverse proxy.

**Implemented Standards:**

Xandikos implements the following RFC standards:

*   RFC 4918/2518 (Core WebDAV) - *Implemented (except LOCK operations)*
*   RFC 4791 (CalDAV) - *Fully implemented*
*   RFC 6352 (CardDAV) - *Fully implemented*
*   RFC 5397 (Current Principal) - *Fully implemented*
*   RFC 3253 (Versioning Extensions) - *Partially implemented*
*   RFC 3744 (Access Control) - *Partially implemented*
*   RFC 5995 (POST to create members) - *Fully implemented*
*   RFC 5689 (Extended MKCOL) - *Fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully implemented*
*   RFC 7953 (Calendar Availability) - *Fully implemented*

**Not Implemented Standards:**

*   RFC 6638 (CalDAV Scheduling Extensions) - *not implemented*
*   RFC 7809 (CalDAV Time Zone Extensions) - *not implemented*
*   RFC 7529 (WebDAV Quota) - *not implemented*
*   RFC 4709 (WebDAV Mount) - *intentionally not implemented*
*   RFC 5546 (iCal iTIP) - *not implemented*
*   RFC 4324 (iCAL CAP) - *not implemented*

For more detailed information on specification compliance, see the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

**Limitations:**

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

**Supported Clients:**

Xandikos is compatible with a wide range of CalDAV and CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   Sogo Connector for Icedove/Thunderbird
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
*   Home Assistant's CalDAV integration
*   pimsync
*   davcli
*   Thunderbird

**Dependencies:**

Xandikos is built with Python 3 and depends on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using `pip` or your system's package manager (e.g., `apt` on Debian/Ubuntu).

**Docker Deployment:**

Xandikos provides a Dockerfile for easy deployment. The image is regularly built and published at ``ghcr.io/jelmer/xandikos``.

*   Use environment variables to configure the Docker image, including ports, data directories, and more.
*   See the `examples/docker-compose.yml` file and the [man page](https://www.xandikos.org/manpage.html) for configuration details.

**Running Xandikos:**

You can run Xandikos directly or behind a reverse proxy like Apache or nginx.

*   To run a standalone instance for testing: `./bin/xandikos --defaults -d $HOME/dav`
*   The server will listen on `http://localhost:8080` by default.

**Client Instructions:**

*   Clients that support RFC 5397 can automatically discover calendar and addressbook URLs.
*   Clients that do not support automatic discovery will need the direct URLs to your calendars/addressbooks.

    Example URLs:

    *   `http://dav.example.com/user/calendars/calendar`
    *   `http://dav.example.com/user/contacts/addressbook`

**Contributing:**

Contributions are welcome!  Please submit issues or pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  New contributor-friendly issues are tagged with the `new-contributor` label.  See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

**Help and Support:**

*   Join the `#xandikos` IRC channel on the OFTC IRC network.
*   Subscribe to the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
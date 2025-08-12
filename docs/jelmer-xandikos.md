<!-- Improved README with SEO Optimization -->

# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a powerful, open-source CalDAV and CardDAV server that stores your calendar and contact data directly in a Git repository, offering robust versioning and data control.** ([View on GitHub](https://github.com/jelmer/xandikos))

![Xandikos Logo](logo.png "Xandikos Logo")

[Extended documentation can be found on the home page <https://www.xandikos.org/docs/>]

## Key Features

*   **Git-backed Storage:** Leverage the power of Git for data versioning, backup, and easy management.
*   **Full CalDAV and CardDAV Support:**  Compatible with a wide range of clients for seamless calendar and contact synchronization.
*   **Lightweight and Efficient:**  Designed for performance, making it ideal for resource-constrained environments.
*   **Open Source:**  Freely available under an open-source license, allowing for community contributions and customization.
*   **Docker Support:** Easily deploy and manage Xandikos using Docker containers.
*   **RFC Compliant:** Adheres to key WebDAV, CalDAV, and CardDAV standards for broad compatibility.

## Standards Implemented

Xandikos implements a wide range of standards for CalDAV and CardDAV compatibility:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - Partial implementation
*   RFC 3744 (Access Control) - Partial implementation
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

## Limitations

*   **Single-User Support:**  Xandikos is currently designed for single-user deployments.
*   **No CalDAV Scheduling Extensions:**  Scheduling extensions are not yet implemented.

## Supported Clients

Xandikos is compatible with a wide array of CalDAV/CardDAV clients, including:

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

Xandikos is built using Python 3 and depends on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

**Installation example using pip:**

```bash
python setup.py develop
```

## Docker

Xandikos offers a Docker image for easy deployment.

**Docker Hub:** `ghcr.io/jelmer/xandikos`

**Configuration:** Configure the Docker image using environment variables such as:

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

See the `examples/docker-compose.yml` and `man page <https://www.xandikos.org/manpage.html>`_ for detailed configuration options.

## Running

Xandikos can be run directly or behind a reverse proxy (e.g., Apache, Nginx).

**Example (standalone, with defaults):**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This will start a server on `http://localhost:8080`.

## Client Instructions

Clients that support RFC:5397 can automatically discover the calendar and addressbook URLs.
For other clients, use the full URL (e.g., `http://dav.example.com/user/calendars/calendar` or `http://dav.example.com/user/contacts/addressbook`).

## Contributing

Contributions are welcome! Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Review the `CONTRIBUTING <CONTRIBUTING.md>`_ guide for more information.

## Help and Support

*   **IRC:** `#xandikos` on the OFTC IRC network.
*   **Mailing List:** [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
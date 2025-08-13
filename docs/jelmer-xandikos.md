# Xandikos: A Lightweight Git-Backed CalDAV/CardDAV Server

**Xandikos** is a complete, lightweight CalDAV and CardDAV server that stores your calendar and contact data using Git, offering a robust and flexible way to manage your personal information. Explore the project on [GitHub](https://github.com/jelmer/xandikos).

![Xandikos Logo](logo.png)

## Key Features

*   **Git-Backed Storage:** Leverages Git for version control and data storage, ensuring data integrity and easy backups.
*   **Comprehensive Standard Support:** Implements a wide range of CalDAV and CardDAV standards for broad client compatibility.
*   **Lightweight & Efficient:** Designed to be fast and resource-friendly, making it suitable for various environments.
*   **Easy to Deploy:**  Available as a Docker image for simple deployment and configuration.
*   **Client Compatibility:** Works with a wide array of CalDAV and CardDAV clients.

## Implemented Standards

Xandikos supports the following standards:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV)
*   RFC 6352 (CardDAV)
*   RFC 5397 (Current Principal)
*   RFC 3253 (Versioning Extensions) - Partial
*   RFC 3744 (Access Control) - Partial
*   RFC 5995 (POST to create members)
*   RFC 5689 (Extended MKCOL)
*   RFC 6578 (Collection Synchronization for WebDAV)
*   RFC 7953 (Calendar Availability)

For more details, see the [DAV compliance notes](https://www.xandikos.org/docs/).

## Supported Clients

Xandikos is compatible with many popular CalDAV/CardDAV clients, including:

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

## Getting Started

### Dependencies

Xandikos supports Python 3 and PyPy 3. It uses the following dependencies:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install these dependencies using pip:

```bash
pip install dulwich jinja2 icalendar defusedxml
```

### Docker

A Dockerfile is provided for easy deployment. The Docker image is available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

### Running

For more details on running the server, please see the full [documentation](https://www.xandikos.org/docs/).

## Contributing

Contributions are welcome! Report bugs, request features, and contribute code or documentation on [GitHub](https://github.com/jelmer/xandikos/issues/new).  See the [CONTRIBUTING](CONTRIBUTING.md) file for details.  New contributor friendly issues are tagged with the `new-contributor` label on GitHub.

## Help and Support

Get help and connect with the community:

*   IRC channel: `#xandikos` on OFTC
*   Mailing list: [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
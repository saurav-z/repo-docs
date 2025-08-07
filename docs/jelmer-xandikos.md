<!--  Xandikos: A Lightweight, Git-Backed CalDAV/CardDAV Server  -->

<div align="center">
  <img src="logo.png" alt="Xandikos Logo" width="200">
</div>

# Xandikos: Your Lightweight, Git-Powered CalDAV/CardDAV Server

**Xandikos** is a self-hosted CalDAV and CardDAV server that stores your calendar and contact data in a Git repository, offering a simple, reliable, and version-controlled solution for syncing your devices. [Explore the Xandikos repository on GitHub](https://github.com/jelmer/xandikos).

## Key Features

*   **Standards Compliance:** Fully implements core WebDAV, CalDAV, and CardDAV standards, ensuring compatibility with a wide range of clients.
*   **Git-Backed Storage:** Utilizes Git for data storage, enabling version control, data integrity, and easy backups.
*   **Lightweight and Efficient:** Designed for simplicity and performance, making it suitable for resource-constrained environments.
*   **Easy to Deploy:** Offers straightforward setup and configuration options, including Docker support.
*   **Client Compatibility:** Works with popular CalDAV/CardDAV clients.

## Implemented Standards

Xandikos supports a wide array of standards, ensuring broad compatibility:

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - *implemented, except for LOCK operations (COPY/MOVE implemented for non-collections)*
*   :RFC:`4791` (CalDAV) - *fully implemented*
*   :RFC:`6352` (CardDAV) - *fully implemented*
*   :RFC:`5397` (Current Principal) - *fully implemented*
*   :RFC:`3253` (Versioning Extensions) - *partially implemented, only the REPORT method and {DAV:}expand-property property*
*   :RFC:`3744` (Access Control) - *partially implemented*
*   :RFC:`5995` (POST to create members) - *fully implemented*
*   :RFC:`5689` (Extended MKCOL) - *fully implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *fully implemented*
*   :RFC:`7953` (Calendar Availability) - *fully implemented*

## Clients Supported

Xandikos is compatible with a variety of CalDAV and CardDAV clients, including:

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

Xandikos is built with:

*   Python 3 (see pyproject.toml for specific version)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

## Docker

Xandikos provides a Docker image for easy deployment. The image is regularly built and published at `ghcr.io/jelmer/xandikos`. Configuration is possible through environment variables.

## Running

Xandikos can be run directly or behind a reverse proxy (e.g., Apache or nginx). Instructions are available in the original repository's documentation.

## Contributing

Contributions are welcome! Please read the `CONTRIBUTING.md` file for guidelines.  Report bugs or suggest features through the [GitHub issues](https://github.com/jelmer/xandikos/issues/new).

## Get Help

*   IRC: #xandikos on OFTC
*   Mailing List: Xandikos Google Group
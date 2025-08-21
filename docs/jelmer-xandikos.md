# Xandikos: Your Lightweight, Git-Backed CalDAV/CardDAV Server

**Xandikos is a powerful and easy-to-use CalDAV/CardDAV server that stores your data in a Git repository, providing version control and data integrity.** ([View on GitHub](https://github.com/jelmer/xandikos))

<p align="center">
  <img src="logo.png" alt="Xandikos Logo" width="200">
</p>

## Key Features

*   **Git-Backed Storage:** Leverage the power of Git for versioning, backups, and data integrity.
*   **CalDAV & CardDAV Support:** Fully compliant with core CalDAV and CardDAV standards.
*   **Lightweight & Fast:** Designed for performance and ease of use.
*   **Easy to Deploy:** Available as a Docker image and supports reverse proxy configurations.
*   **Open Source:** Contribute to the project and customize it to your needs.

## Implemented Standards

Xandikos supports a wide range of WebDAV, CalDAV, and CardDAV standards:

*   RFC 4918/2518 (Core WebDAV) - *implemented, except for LOCK operations*
*   RFC 4791 (CalDAV) - *fully implemented*
*   RFC 6352 (CardDAV) - *fully implemented*
*   RFC 5397 (Current Principal) - *fully implemented*
*   RFC 3253 (Versioning Extensions) - *partially implemented*
*   RFC 3744 (Access Control) - *partially implemented*
*   RFC 5995 (POST to create members) - *fully implemented*
*   RFC 5689 (Extended MKCOL) - *fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *fully implemented*
*   RFC 7953 (Calendar Availability) - *fully implemented*

See [DAV compliance](https://www.xandikos.org/docs/) for more detailed information on specification compliance.

## Supported Clients

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

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy. For detailed instructions, refer to the [documentation](https://www.xandikos.org/docs/).

### Docker

A Dockerfile is provided, with a regularly built and published image available at `ghcr.io/jelmer/xandikos`.

*   **Pull the latest image:** `docker pull ghcr.io/jelmer/xandikos`
*   **Configuration:** Use environment variables to customize your deployment (PORT, DATA_DIR, etc.). See the [docker-compose.yml](https://github.com/jelmer/xandikos/blob/master/examples/docker-compose.yml) and [man page](https://www.xandikos.org/manpage.html) for details.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](https://github.com/jelmer/xandikos/blob/master/CONTRIBUTING.md) file for guidelines.  Report bugs and request features on [GitHub Issues](https://github.com/jelmer/xandikos/issues/new).

## Get Help

*   **IRC:** `#xandikos` on the OFTC IRC network.
*   **Mailing List:** [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
# Xandikos: Your Lightweight Git-Backed CalDAV/CardDAV Server

**Xandikos is a complete, lightweight CalDAV/CardDAV server that stores your calendar and contact data in a Git repository, providing a flexible and reliable way to manage your data.**  [View the original repository on GitHub](https://github.com/jelmer/xandikos)

[![Xandikos Logo](logo.png)](https://github.com/jelmer/xandikos)

## Key Features

*   **Git-Based Storage:** All data is stored in a Git repository for version control, backup, and easy access.
*   **Full CalDAV & CardDAV Support:** Supports core RFCs for calendar and contact data synchronization.
*   **Lightweight & Efficient:** Designed for low resource usage.
*   **Docker Support:** Easily deployable with a pre-built Docker image.
*   **Open Source:** Free and open-source software, available under the MIT license.

## Implemented Standards

Xandikos implements a wide range of WebDAV, CalDAV, and CardDAV RFCs, ensuring compatibility with various clients:

*   RFC 4918/2518 (Core WebDAV) - *Implemented (except LOCK operations)*
*   RFC 4791 (CalDAV) - *Fully Implemented*
*   RFC 6352 (CardDAV) - *Fully Implemented*
*   RFC 5397 (Current Principal) - *Fully Implemented*
*   RFC 3253 (Versioning Extensions) - *Partially Implemented*
*   RFC 3744 (Access Control) - *Partially Implemented*
*   RFC 5995 (POST to create members) - *Fully Implemented*
*   RFC 5689 (Extended MKCOL) - *Fully Implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully Implemented*
*   RFC 7953 (Calendar Availability) - *Fully Implemented*

For detailed compliance information, see the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

## Supported Clients

Xandikos is compatible with a wide range of CalDAV and CardDAV clients, including:

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
*   Home Assistant's CalDAV integration
*   pimsync
*   davcli
*   Thunderbird

## Getting Started

### Dependencies

Xandikos requires Python 3 (see `pyproject.toml` for specific versions), Pypy 3 and the following Python packages: Dulwich, Jinja2, icalendar, and defusedxml.

You can install these dependencies using `pip`:

```bash
python setup.py develop
```

or

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

### Running

To run a standalone (no authentication) instance with a pre-created calendar and addressbook:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

A server will then be listening on `http://localhost:8080`.

### Docker

A Dockerfile is provided for easy deployment. You can find the latest image on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos). Use environment variables to configure the container.

## Contributing

Contributions are welcome! Report bugs and suggest features on [GitHub Issues](https://github.com/jelmer/xandikos/issues/new). See the [CONTRIBUTING](CONTRIBUTING.md) file for guidelines.

## Get Help

*   **IRC:**  #xandikos on OFTC
*   **Mailing List:**  [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
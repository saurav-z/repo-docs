<!-- Improved & Summarized README for Xandikos -->
# Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git

**Xandikos is a simple yet powerful server that allows you to host your calendars and contacts, offering a unique approach by storing your data in a Git repository.** ([Original Repo](https://github.com/jelmer/xandikos))

![Xandikos Logo](logo.png)

Find in-depth documentation on the [Xandikos website](https://www.xandikos.org/docs/).

## Key Features

*   **Git-backed Storage:** Uses a Git repository for version control and data storage, offering data durability and easy backups.
*   **CalDAV & CardDAV Compliance:** Implements a wide range of CalDAV (calendars) and CardDAV (contacts) standards for compatibility with various clients.
*   **Lightweight and Easy to Deploy:** Designed to be lightweight and straightforward to set up, with Docker support for containerization.
*   **Client Compatibility:** Works with a variety of popular CalDAV and CardDAV clients.
*   **Open Source:** MIT licensed

## Implemented RFC Standards

Xandikos offers robust support for core WebDAV and the following CalDAV and CardDAV standards:

*   RFC 4918/2518 (Core WebDAV)
*   RFC 4791 (CalDAV) - *Fully Implemented*
*   RFC 6352 (CardDAV) - *Fully Implemented*
*   RFC 5397 (Current Principal) - *Fully Implemented*
*   RFC 3253 (Versioning Extensions) - *Partially Implemented*
*   RFC 3744 (Access Control) - *Partially Implemented*
*   RFC 5995 (POST to create members) - *Fully Implemented*
*   RFC 5689 (Extended MKCOL) - *Fully Implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully Implemented*
*   RFC 7953 (Calendar Availability) - *Fully Implemented*

## Supported Clients

Xandikos has been tested with and is compatible with numerous CalDAV/CardDAV clients, including but not limited to:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   Sogo Connector for Icedove/Thunderbird
*   aCALdav syncer for Android
*   pycardsyncer
*   Akonadi
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

## Installation and Usage

Xandikos can be run directly or behind a reverse proxy (recommended for production). Docker images are also available.

### Dependencies

Xandikos requires:

*   Python 3 (See `pyproject.toml` for the specific version)
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip:

```bash
python setup.py develop
```

### Docker

The Docker image is regularly built and available at `ghcr.io/jelmer/xandikos`.

For each release, a `v$RELEASE` tag is available - e.g. `v0.2.11` for release *0.2.11*.
For a full list, see [the Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

The Docker image can be configured using environment variables (see original README for complete list).

### Running

To run a standalone instance with a pre-created calendar and addressbook:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

## Contributing

Contributions are welcome! Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new). See `CONTRIBUTING.md` for details.

## Get Help

*   IRC: `#xandikos` on OFTC.
*   Mailing List: [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
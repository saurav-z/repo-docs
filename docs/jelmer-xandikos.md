<!-- SEO-optimized README for Xandikos -->

# Xandikos: A Lightweight, Git-Backed CalDAV/CardDAV Server

**Xandikos empowers you to manage your calendars and contacts with the power of Git, offering a simple yet robust CalDAV/CardDAV server.**  [View the original repo](https://github.com/jelmer/xandikos).

![Xandikos Logo](logo.png)

Xandikos (Ξανδικός or Ξανθικός) is named after the March month in the ancient Macedonian calendar. Explore comprehensive documentation on the [Xandikos home page](https://www.xandikos.org/docs/).

## Key Features

*   **Git-Backed Storage:** Utilizes a Git repository for data storage, enabling versioning, backups, and easy management of your calendar and contact data.
*   **CalDAV & CardDAV Compliant:** Fully implements core CalDAV (RFC 4791) and CardDAV (RFC 6352) standards, ensuring compatibility with a wide range of clients.
*   **Lightweight & Efficient:** Designed for performance and ease of use, perfect for personal or small-team deployments.
*   **Docker Support:**  Easily deploy Xandikos using Docker for simplified setup and management.
*   **Open Source:**  Benefit from the transparency and flexibility of an open-source solution.

## Implemented Standards

Xandikos supports a wide range of WebDAV, CalDAV, and CardDAV standards:

*   RFC 4918/2518 (Core WebDAV) - *Implemented*
*   RFC 4791 (CalDAV) - *Fully implemented*
*   RFC 6352 (CardDAV) - *Fully implemented*
*   RFC 5397 (Current Principal) - *Fully implemented*
*   RFC 3253 (Versioning Extensions) - *Partially implemented*
*   RFC 3744 (Access Control) - *Partially implemented*
*   RFC 5995 (POST to create members) - *Fully implemented*
*   RFC 5689 (Extended MKCOL) - *Fully implemented*
*   RFC 6578 (Collection Synchronization for WebDAV) - *Fully implemented*
*   RFC 7953 (Calendar Availability) - *Fully implemented*

## Limitations

*   No multi-user support.
*   No support for CalDAV scheduling extensions.

## Supported Clients

Xandikos is compatible with various CalDAV and CardDAV clients, including:

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
<!-- Add a few more popular clients to broaden the target -->

## Dependencies

Xandikos is built using Python 3 and depends on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip: `python setup.py develop` or apt: `sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2`

## Docker

Xandikos offers a Docker image for easy deployment, available on [GitHub Container Registry](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

Configure the Docker image using environment variables.  See the Dockerfile and `examples/docker-compose.yml` for details.

## Running Xandikos

Xandikos can be run directly or behind a reverse proxy.  To run a standalone instance with defaults:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

The server will listen on `localhost:8000`

## Client Configuration

Clients supporting RFC:5397 can use the base URL. For clients without auto-discovery, use specific URLs like:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

## Contributing

Contributions are welcome!  Report bugs, request features, and contribute code via [GitHub Issues](https://github.com/jelmer/xandikos/issues/new). Review the [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## Help & Support

*   IRC:  #xandikos on the OFTC IRC network.
*   Mailing List:  [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
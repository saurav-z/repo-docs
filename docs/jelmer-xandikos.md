<div align="center">
  <img src="logo.png" alt="Xandikos Logo" width="200">
</div>

# Xandikos: Your Lightweight, Git-Backed CardDAV/CalDAV Server

**Xandikos is a fast and reliable CardDAV/CalDAV server that stores your calendar and contact data in a Git repository.**  Check out the original repo [here](https://github.com/jelmer/xandikos).

## Key Features

*   **Git-Backed Storage:**  Leverages Git for version control and data persistence, ensuring data integrity and easy backups.
*   **Complete Standard Compliance:** Fully implements major CalDAV and CardDAV RFCs.
*   **Lightweight and Fast:** Designed for performance and ease of use.
*   **Docker Support:** Easily deployable with Docker for containerized environments.
*   **Multiple Client Compatibility:** Works seamlessly with a wide range of CalDAV/CardDAV clients.

## Implemented Standards

Xandikos supports a comprehensive set of WebDAV, CalDAV, and CardDAV standards.  Here's a summary:

*   **Core WebDAV:** :RFC:`4918`/:RFC:`2518` (Implemented, except for LOCK operations - COPY/MOVE implemented for non-collections)
*   **CalDAV:** :RFC:`4791` (Fully Implemented)
*   **CardDAV:** :RFC:`6352` (Fully Implemented)
*   **Current Principal:** :RFC:`5397` (Fully Implemented)
*   **Versioning Extensions:** :RFC:`3253` (Partially Implemented)
*   **Access Control:** :RFC:`3744` (Partially Implemented)
*   **POST to Create Members:** :RFC:`5995` (Fully Implemented)
*   **Extended MKCOL:** :RFC:`5689` (Fully Implemented)
*   **Collection Synchronization for WebDAV:** :RFC:`6578` (Fully Implemented)
*   **Calendar Availability:** :RFC:`7953` (Fully Implemented)

## Limitations

*   **Single-User Support:**  Xandikos currently only supports a single user.
*   **No Scheduling Extensions:** CalDAV scheduling extensions are not implemented.

## Supported Clients

Xandikos is compatible with a wide variety of CalDAV/CardDAV clients, including:

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

## Installation and Deployment

### Dependencies

Xandikos requires Python 3 (see `pyproject.toml` for specific version) and utilizes several Python libraries: Dulwich, Jinja2, icalendar, and defusedxml.

### Docker

A Dockerfile is provided for easy deployment. The Docker image is available at `ghcr.io/jelmer/xandikos`.

*   **Environment Variables:** Configure the Docker image using environment variables for port, data directory, user principal, and other settings. See the original README for all variables, and `examples/docker-compose.yml` for an example.

### Running Locally

To run a standalone instance, use:

```bash
./bin/xandikos --defaults -d $HOME/dav
```

This creates a server listening on `localhost:8080` with a pre-created calendar and addressbook.

### Production

For production environments, it's recommended to run Xandikos behind a reverse HTTP proxy like Apache or nginx.

## Contributing

Contributions are welcome!  Please submit bug reports and feature requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Read `CONTRIBUTING.md` for information on contributing code and documentation.  Look for issues tagged `new-contributor` for good starting points.

## Getting Help

*   **IRC:** Join the *#xandikos* channel on the `OFTC <https://www.oftc.net/>` IRC network.
*   **Mailing List:** Subscribe to the `Xandikos <https://groups.google.com/forum/#!forum/xandikos>` mailing list.
*   **Documentation:** Detailed documentation can be found `on the home page <https://www.xandikos.org/docs/>`_.
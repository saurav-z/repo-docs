<!-- Xandikos: Lightweight CalDAV/CardDAV Server Backed by Git -->

<!-- Header with Logo -->
<div align="center">
  <a href="https://github.com/jelmer/xandikos">
    <img src="logo.png" alt="Xandikos Logo" width="200">
  </a>
</div>

# Xandikos: Your Git-Powered CalDAV/CardDAV Server

**Xandikos** is a lightweight, yet powerful CalDAV/CardDAV server that stores your calendar and contact data directly in a Git repository, providing robust versioning and easy management. [View the original repo](https://github.com/jelmer/xandikos).

## Key Features

*   **Git-Backed Storage:** Leverage the power of Git for version control, backups, and data integrity.
*   **Complete Protocol Support:** Implements core WebDAV, CalDAV, and CardDAV standards.
*   **Lightweight and Efficient:** Designed for ease of use and minimal resource consumption.
*   **Docker Support:** Easy deployment with pre-built Docker images.
*   **Client Compatibility:** Works with a wide range of CalDAV/CardDAV clients.

## Implemented Standards

Xandikos supports a comprehensive set of CalDAV and CardDAV standards:

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

For detailed compliance information, refer to the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

## Limitations

*   **No Multi-User Support:** Xandikos is designed for single-user deployments.
*   **No CalDAV Scheduling Extensions:** Does not support CalDAV scheduling features.

## Supported Clients

Xandikos works seamlessly with a variety of CalDAV/CardDAV clients:

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

Xandikos requires:

*   Python 3 (see `pyproject.toml` for specific version)
*   Pypy 3
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies using pip:

```bash
python setup.py develop
```

### Docker

Build and run a Docker container:

1.  **Build the Image:**
    ```bash
    docker build -t xandikos .
    ```
2.  **Run the Container:**
    ```bash
    docker run -d -p 8000:8000 -v /path/to/data:/data xandikos
    ```

    Customize the environment variables in the `docker run` command to configure the server (e.g. `PORT`, `DATA_DIR`, `DEBUG`). Refer to the [Docker documentation](https://www.xandikos.org/docs/docker/) for details.

### Running Directly

1.  **Create a Data Directory:**
    ```bash
    mkdir -p $HOME/dav
    ```
2.  **Run the Server:**
    ```bash
    ./bin/xandikos --defaults -d $HOME/dav
    ```

    This creates a basic setup with a calendar and address book. Access the server at `http://localhost:8080/`.

## Configuration Options

Xandikos can be configured through command-line arguments and environment variables (when using Docker). Key options include:

*   `--defaults`: Creates default calendar and address book.
*   `-d <data_dir>`: Specifies the data directory.
*   `-p <port>`: Sets the listening port.

## Contributing

We welcome contributions! If you find a bug or have a feature request, please open an issue on [GitHub](https://github.com/jelmer/xandikos/issues/new). For code or documentation contributions, please review the [CONTRIBUTING](CONTRIBUTING.md) guidelines.

## Help and Support

*   **IRC:** Join the *#xandikos* channel on the [OFTC](https://www.oftc.net/) IRC network.
*   **Mailing List:** Subscribe to the [Xandikos mailing list](https://groups.google.com/forum/#!forum/xandikos).
*   **Documentation:** Visit the [Xandikos documentation](https://www.xandikos.org/docs/) for detailed information.
<div align="center">
  <img src="logo.png" alt="Xandikos Logo" width="200">
</div>

# Xandikos: Your Lightweight, Git-Backed CardDAV/CalDAV Server

**Xandikos** is a powerful yet streamlined server for synchronizing your calendars and contacts, leveraging the version control capabilities of Git. [Learn more on GitHub](https://github.com/jelmer/xandikos).

**Key Features:**

*   🔄 **CardDAV/CalDAV Synchronization:** Supports standard protocols for seamless calendar and contact syncing.
*   💾 **Git-Based Storage:**  Utilizes a Git repository as its backend, providing version control and data durability.
*   📦 **Lightweight & Efficient:** Designed for performance and ease of use.
*   ⚙️ **Easy to Deploy:** Available as a Docker image for simple setup.
*   🤝 **Client Compatibility:** Works with a wide range of popular CalDAV/CardDAV clients.
*   💻 **Open Source:**  Free and open-source under the MIT license.

## Implemented Standards

Xandikos supports the following standards:

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

**Not Implemented:**

*   RFC 6638 (CalDAV Scheduling Extensions)
*   RFC 7809 (CalDAV Time Zone Extensions)
*   RFC 7529 (WebDAV Quota)
*   RFC 4709 (WebDAV Mount) - *Intentionally not implemented*
*   RFC 5546 (iCal iTIP)
*   RFC 4324 (iCAL CAP)

For more detailed information on specification compliance, see the [DAV compliance notes](https://www.xandikos.org/docs/dav-compliance.html).

## Limitations

*   No multi-user support
*   No support for CalDAV scheduling extensions

## Supported Clients

Xandikos works with the following CalDAV/CardDAV clients:

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

Xandikos is built using:

*   Python 3 (see pyproject.toml for supported versions)
*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

**Installation (Debian/Ubuntu):**

```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

**Installation (using pip):**

```bash
python setup.py develop
```

## Docker

Xandikos provides a Docker image for easy deployment, regularly built and published at `ghcr.io/jelmer/xandikos`.  Use the `v$RELEASE` tags (e.g., `v0.2.11`).  For a complete list, visit the [Container overview page](https://github.com/jelmer/xandikos/pkgs/container/xandikos).

**Configuration via Environment Variables:**

*   `PORT`:  Port to listen on (default: 8000)
*   `METRICS_PORT`: Port for metrics endpoint (default: 8001)
*   `LISTEN_ADDRESS`: Address to bind to (default: 0.0.0.0)
*   `DATA_DIR`: Data directory path (default: /data)
*   `CURRENT_USER_PRINCIPAL`: User principal path (default: /user/)
*   `ROUTE_PREFIX`: URL route prefix (default: /)
*   `AUTOCREATE`: Auto-create directories (true/false)
*   `DEFAULTS`: Create default calendar/addressbook (true/false)
*   `DEBUG`: Enable debug logging (true/false)
*   `DUMP_DAV_XML`: Print DAV XML requests/responses (true/false)
*   `NO_STRICT`: Enable client compatibility workarounds (true/false)

See `examples/docker-compose.yml` and the [man page](https://www.xandikos.org/manpage.html) for detailed configuration instructions.

## Running

Xandikos can run directly on HTTP or behind a reverse proxy like Apache or nginx.

**Testing:**

```bash
./bin/xandikos --defaults -d $HOME/dav
```

Access it at `http://localhost:8080/`.

**Production:**

Use a reverse proxy (Apache, nginx) for production deployments. Example init system configurations are in the `examples/` directory.

## Client Instructions

Clients with auto-discovery (RFC 5397 support) can use the base URL.  Clients without auto-discovery require the direct URL to a calendar or address book.  If you used the `--defaults` argument, the URLs will be:

*   `http://dav.example.com/user/calendars/calendar`
*   `http://dav.example.com/user/contacts/addressbook`

## Contributing

Contributions are welcome!  Report bugs and request features on [GitHub](https://github.com/jelmer/xandikos/issues/new).  Read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code and documentation contributions. Issues for new contributors are tagged `new-contributor`.

## Help

*   IRC: `#xandikos` on OFTC ([https://www.oftc.net/](https://www.oftc.net/))
*   Mailing List: [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
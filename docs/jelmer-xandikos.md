<!-- Improved & SEO-Optimized README for Xandikos -->

# Xandikos: Your Lightweight Git-Backed CalDAV/CardDAV Server

**Xandikos is a powerful, lightweight, and complete CalDAV/CardDAV server that stores your calendar and contact data in a Git repository, offering version control and data integrity.**

[Visit the original repository on GitHub](https://github.com/jelmer/xandikos)

<!-- Logo (Optional - consider adding alt text for SEO) -->
<p align="center">
  <img src="logo.png" alt="Xandikos Logo" width="200">
</p>

**Key Features:**

*   **Git-Based Storage:** Leverage the power of Git for versioning, backups, and data integrity.
*   **Full CalDAV and CardDAV Support:**  Compatible with a wide range of clients, offering complete support for calendar and contact synchronization.
*   **Lightweight and Efficient:** Designed to be resource-friendly, ideal for personal use or small deployments.
*   **Standards Compliant:** Implements core WebDAV, CalDAV, CardDAV, and other key RFCs (see below).
*   **Easy to Deploy:**  Runs directly or behind a reverse proxy, and Docker support is available.

**Implemented Standards:**

*   :RFC:`4918`/:RFC:`2518` (Core WebDAV) - (Implemented, except for LOCK operations)
*   :RFC:`4791` (CalDAV) - *Fully Implemented*
*   :RFC:`6352` (CardDAV) - *Fully Implemented*
*   :RFC:`5397` (Current Principal) - *Fully Implemented*
*   :RFC:`3253` (Versioning Extensions) - *Partially Implemented*
*   :RFC:`3744` (Access Control) - *Partially Implemented*
*   :RFC:`5995` (POST to create members) - *Fully Implemented*
*   :RFC:`5689` (Extended MKCOL) - *Fully Implemented*
*   :RFC:`6578` (Collection Synchronization for WebDAV) - *Fully Implemented*
*   :RFC:`7953` (Calendar Availability) - *Fully Implemented*

**Not Implemented Standards:**

*   :RFC:`6638` (CalDAV Scheduling Extensions) - *Not Implemented*
*   :RFC:`7809` (CalDAV Time Zone Extensions) - *Not Implemented*
*   :RFC:`7529` (WebDAV Quota) - *Not Implemented*
*   :RFC:`4709` (WebDAV Mount) - *Not Implemented*
*   :RFC:`5546` (iCal iTIP) - *Not Implemented*
*   :RFC:`4324` (iCAL CAP) - *Not Implemented*

For more details on specification compliance, see the `DAV compliance <notes/dav-compliance.rst>`_ documentation.

**Limitations:**

*   No multi-user support (single-user focus).
*   No support for CalDAV scheduling extensions.

**Supported Clients:**

Xandikos is compatible with a wide array of CalDAV/CardDAV clients, including:

*   Vdirsyncer
*   caldavzap/carddavmate
*   Evolution
*   DAVx5 (formerly DAVDroid)
*   Sogo connector for Icedove/Thunderbird
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

**Dependencies:**

Xandikos is a Python 3 application, relying on the following libraries:

*   Dulwich
*   Jinja2
*   icalendar
*   defusedxml

Install dependencies on Debian:
```bash
sudo apt install python3-dulwich python3-defusedxml python3-icalendar python3-jinja2
```

Or install using pip:
```bash
python setup.py develop
```

**Docker Support:**

A Dockerfile is included for easy containerization.  The image is available on [ghcr.io/jelmer/xandikos](https://github.com/jelmer/xandikos/pkgs/container/xandikos) and tagged with releases like `v0.2.11`.

Configure the Docker image using environment variables such as `PORT`, `DATA_DIR`, `DEBUG`, and others. See the documentation for a complete list.

**Running Xandikos:**

You can run Xandikos directly or behind a reverse proxy.

To run a standalone instance for testing:
```bash
./bin/xandikos --defaults -d $HOME/dav
```
This will create a calendar and addressbook in *$HOME/dav* and listen on `localhost:8080 <http://localhost:8080/>`_.

**Client Configuration:**

Clients supporting RFC:`5397` can auto-discover the service.  Otherwise, you'll need the full URLs:

```
http://dav.example.com/user/calendars/calendar
http://dav.example.com/user/contacts/addressbook
```

**Contributing:**

Contributions are welcome!  Please submit issues and pull requests on [GitHub](https://github.com/jelmer/xandikos/issues/new).  See `CONTRIBUTING.md` for more details.  Look for the `new-contributor` tag for beginner-friendly issues.

**Help and Support:**

*   IRC:  *#xandikos* on the OFTC IRC network.
*   Mailing List:  [Xandikos Google Group](https://groups.google.com/forum/#!forum/xandikos)
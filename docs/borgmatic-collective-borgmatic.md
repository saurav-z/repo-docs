# borgmatic: Secure, Configuration-Driven Backups for Servers and Workstations

Safeguard your valuable data with borgmatic, the easy-to-configure backup solution. Visit the [borgmatic GitHub repository](https://github.com/borgmatic-collective/borgmatic) for more information.

<img src="docs/static/borgmatic.png" alt="borgmatic logo" width="150px" style="float: right; padding-left: 1em;">

**Key Features:**

*   **Client-Side Encryption:** Protect your data with robust encryption before it leaves your system.
*   **Configuration-Driven:**  Define your backup strategy using simple, easy-to-understand configuration files.
*   **Database Backup Support:** Back up popular databases like PostgreSQL, MySQL, and more.
*   **Flexible Repository Options:** Backup to local storage, remote servers via SSH, or cloud storage providers.
*   **Retention Policies:** Easily manage your backups with customizable retention settings (daily, weekly, monthly).
*   **Backup Verification:**  Regularly check your backups to ensure data integrity.
*   **Monitoring & Notifications:** Integrate with third-party services like Healthchecks.io, Uptime Kuma, and others to monitor your backup status.
*   **Modular Design:** Easily integrates with Docker, Podman, and systemd for flexible management.

**Integrations:**

borgmatic seamlessly integrates with a wide range of services, including:

### Data

<a href="https://www.postgresql.org/"><img src="docs/static/postgresql.png" alt="PostgreSQL" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.mysql.com/"><img src="docs/static/mysql.png" alt="MySQL" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://mariadb.com/"><img src="docs/static/mariadb.png" alt="MariaDB" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.mongodb.com/"><img src="docs/static/mongodb.png" alt="MongoDB" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://sqlite.org/"><img src="docs/static/sqlite.png" alt="SQLite" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://openzfs.org/"><img src="docs/static/openzfs.png" alt="OpenZFS" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://btrfs.readthedocs.io/"><img src="docs/static/btrfs.png" alt="Btrfs" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://sourceware.org/lvm2/"><img src="docs/static/lvm.png" alt="LVM" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://rclone.org"><img src="docs/static/rclone.png" alt="rclone" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.borgbase.com/?utm_source=borgmatic"><img src="docs/static/borgbase.png" alt="BorgBase" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>

### Monitoring

<a href="https://healthchecks.io/"><img src="docs/static/healthchecks.png" alt="Healthchecks" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://uptime.kuma.pet/"><img src="docs/static/uptimekuma.png" alt="Uptime Kuma" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://cronitor.io/"><img src="docs/static/cronitor.png" alt="Cronitor" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://cronhub.io/"><img src="docs/static/cronhub.png" alt="Cronhub" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.pagerduty.com/"><img src="docs/static/pagerduty.png" alt="PagerDuty" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.pushover.net/"><img src="docs/static/pushover.png" alt="Pushover" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://ntfy.sh/"><img src="docs/static/ntfy.png" alt="ntfy" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://grafana.com/oss/loki/"><img src="docs/static/loki.png" alt="Loki" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://github.com/caronc/apprise/wiki"><img src="docs/static/apprise.png" alt="Apprise" height="60px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.zabbix.com/"><img src="docs/static/zabbix.png" alt="Zabbix" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://sentry.io/"><img src="docs/static/sentry.png" alt="Sentry" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>

### Credentials

<a href="https://systemd.io/"><img src="docs/static/systemd.png" alt="Sentry" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://www.docker.com/"><img src="docs/static/docker.png" alt="Docker" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://podman.io/"><img src="docs/static/podman.png" alt="Podman" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>
<a href="https://keepassxc.org/"><img src="docs/static/keepassxc.png" alt="Podman" height="40px" style="margin-bottom:20px; margin-right:20px;"></a>

**Getting Started**

To begin, consult the [borgmatic documentation](https://torsion.org/borgmatic/docs/how-to/set-up-backups/) for detailed installation and configuration instructions.

**Hosting Providers**

Consider these providers for secure, off-site storage of your backups. Note that using these links supports the development of borgmatic:

*   [BorgBase](https://www.borgbase.com/?utm_source=borgmatic): A dedicated Borg hosting service.
*   [Hetzner](https://hetzner.cloud/?ref=v9dOJ98Ic9I8): Hetzner offers a "storage box" with Borg support.

**Support and Community**

*   **Issues:** Report bugs and suggest enhancements via the [issue tracker](https://projects.torsion.org/borgmatic-collective/borgmatic/issues).
*   **Social:** Follow borgmatic on Mastodon: <a rel="me" href="https://floss.social/@borgmatic">@borgmatic@floss.social</a>.
*   **Chat:** Join the `#borgmatic` IRC channel on Libera Chat ([web chat](https://web.libera.chat/#borgmatic), or use an [IRC client](ircs://irc.libera.chat:6697)).
*   **Contact:** For other inquiries, reach out to [witten@torsion.org](mailto:witten@torsion.org).

**Contributing**

borgmatic is an open-source project; your contributions are welcome!

*   **Source Code:** Access the [source code](https://projects.torsion.org/borgmatic-collective/borgmatic) and a read-only mirror on [GitHub](https://github.com/borgmatic-collective/borgmatic).
*   **License:** borgmatic is licensed under the GNU General Public License version 3 or later.
*   **Contribute:** Submit [pull requests](https://projects.torsion.org/borgmatic-collective/borgmatic/pulls) or open [issues](https://projects.torsion.org/borgmatic-collective/borgmatic/issues).

**Recent Contributors**

[Include the contributors.html file here - kept for the sake of the original README]</br>
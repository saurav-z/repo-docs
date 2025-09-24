# Borgmatic: Effortless, Configurable Backups for Servers and Workstations

**Secure your data with borgmatic, a powerful and easy-to-use backup solution that protects your files and databases with client-side encryption.**  [Learn more at the original repository](https://github.com/borgmatic-collective/borgmatic).

## Key Features

*   **Configuration-Driven:** Easily manage your backups through a simple configuration file.
*   **Client-Side Encryption:** Protect your data with robust encryption, ensuring your backups are secure.
*   **Database Support:** Back up your databases (PostgreSQL, MySQL, MariaDB, MongoDB, SQLite) alongside your files.
*   **Flexible Retention Policies:** Define how many backups to keep with customizable retention settings.
*   **Integrated Monitoring:** Monitor your backups with integrations for various third-party services.
*   **Automated Pre/Post Backup Scripts:** Allows the running of custom scripts to prepare data for backup.

## Why Choose Borgmatic?

*   **Simplicity:** Easy to set up and configure, making it accessible for all users.
*   **Security:** Client-side encryption ensures your data is protected at all times.
*   **Reliability:** Built upon the solid foundation of [Borg Backup](https://www.borgbackup.org/), a well-established backup solution.
*   **Flexibility:** Supports a wide range of data sources and destinations.
*   **Monitoring and Notification:** Stay informed about the status of your backups with comprehensive monitoring features.

## Integrations

borgmatic seamlessly integrates with various services to enhance your backup experience:

### Data
*   [PostgreSQL](https://www.postgresql.org/)
*   [MySQL](https://www.mysql.com/)
*   [MariaDB](https://mariadb.com/)
*   [MongoDB](https://www.mongodb.com/)
*   [SQLite](https://sqlite.org/)
*   [OpenZFS](https://openzfs.org/)
*   [Btrfs](https://btrfs.readthedocs.io/)
*   [LVM](https://sourceware.org/lvm2/)
*   [rclone](https://rclone.org)
*   [BorgBase](https://www.borgbase.com/?utm_source=borgmatic)

### Monitoring
*   [Healthchecks](https://healthchecks.io/)
*   [Uptime Kuma](https://uptime.kuma.pet/)
*   [Cronitor](https://cronitor.io/)
*   [Cronhub](https://cronhub.io/)
*   [PagerDuty](https://www.pagerduty.com/)
*   [Pushover](https://www.pushover.net/)
*   [ntfy](https://ntfy.sh/)
*   [Loki](https://grafana.com/oss/loki/)
*   [Apprise](https://github.com/caronc/apprise/wiki)
*   [Zabbix](https://www.zabbix.com/)
*   [Sentry](https://sentry.io/)

### Credentials
*   [systemd](https://systemd.io/)
*   [Docker](https://www.docker.com/)
*   [Podman](https://podman.io/)
*   [KeePassXC](https://keepassxc.org/)

## Getting Started

To start backing up your data with borgmatic, begin by [installing and configuring](https://torsion.org/borgmatic/docs/how-to/set-up-backups/) the software.

## Hosting Providers

For off-site backup storage, consider these providers that support Borg/borgmatic:

*   <a href="https://www.borgbase.com/?utm_source=borgmatic">BorgBase</a>: Borg hosting with monitoring, 2FA, and append-only repos.
*   <a href="https://hetzner.cloud/?ref=v9dOJ98Ic9I8">Hetzner</a>: Storage boxes that support Borg.

rsync.net also offers compatible storage.

## Support and Contributing

### Issues
Report issues or suggest feature enhancements on our [issue tracker](https://projects.torsion.org/borgmatic-collective/borgmatic/issues). You'll need to [register](https://projects.torsion.org/user/sign_up?invite_code=borgmatic) to create a new issue or comment, or [login directly](https://projects.torsion.org/user/login) via your GitHub account. See the [security policy](https://torsion.org/borgmatic/docs/security-policy/) for any security issues.

### Social
Follow borgmatic on Mastodon: <a rel="me" href="https://floss.social/@borgmatic">@borgmatic</a>

### Chat
Join the `#borgmatic` IRC channel on Libera Chat via <a href="https://web.libera.chat/#borgmatic">web chat</a> or your favorite <a href="ircs://irc.libera.chat:6697">IRC client</a>.

### Other
Contact [witten@torsion.org](mailto:witten@torsion.org) for other inquiries.

### Contributing
View borgmatic's [source code](https://projects.torsion.org/borgmatic-collective/borgmatic) and its read-only mirror on [GitHub](https://github.com/borgmatic-collective/borgmatic).

borgmatic is licensed under the GNU General Public License version 3 or any later version. To contribute, submit a [pull request](https://projects.torsion.org/borgmatic-collective/borgmatic/pulls) or open an [issue](https://projects.torsion.org/borgmatic-collective/borgmatic/issues), after [registering](https://projects.torsion.org/user/sign_up?invite_code=borgmatic).  Refer to the [borgmatic development how-to](https://torsion.org/borgmatic/docs/how-to/develop-on-borgmatic/) to learn about cloning the source code, running tests, and more.

### Recent Contributors
(Include contributor list here - as per original README)
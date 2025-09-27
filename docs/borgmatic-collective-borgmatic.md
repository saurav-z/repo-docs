# borgmatic: Simple, Configuration-Driven Backup Software

Safeguard your valuable data with **borgmatic**, the easy-to-use backup solution that prioritizes client-side encryption and flexible configuration. For more details, visit the [original borgmatic repository](https://github.com/borgmatic-collective/borgmatic).

![borgmatic logo](docs/static/borgmatic.png)

## Key Features

*   **Configuration-Driven:** Define your backup strategy using a straightforward configuration file.
*   **Client-Side Encryption:** Protect your data with robust client-side encryption.
*   **Database Backup:** Seamlessly back up popular databases like PostgreSQL, MySQL, and more.
*   **Flexible Retention Policies:** Customize how many backups to keep based on your needs (daily, weekly, monthly).
*   **Third-Party Integrations:** Monitor your backups and receive notifications using services like Healthchecks.io.
*   **Supports Rclone:** Backup to many cloud providers using rclone.

## Integrations

borgmatic integrates with a wide range of services for data backup, monitoring, and credential management:

### Data

[![PostgreSQL](docs/static/postgresql.png)](https://www.postgresql.org/)
[![MySQL](docs/static/mysql.png)](docs/static/mysql.png)
[![MariaDB](docs/static/mariadb.png)](https://mariadb.com/)
[![MongoDB](docs/static/mongodb.png)](https://www.mongodb.com/)
[![SQLite](docs/static/sqlite.png)](https://sqlite.org/)
[![OpenZFS](docs/static/openzfs.png)](https://openzfs.org/)
[![Btrfs](docs/static/btrfs.png)](https://btrfs.readthedocs.io/)
[![LVM](docs/static/lvm.png)](https://sourceware.org/lvm2/)
[![rclone](docs/static/rclone.png)](https://rclone.org)
[![BorgBase](docs/static/borgbase.png)](https://www.borgbase.com/?utm_source=borgmatic)

### Monitoring

[![Healthchecks](docs/static/healthchecks.png)](https://healthchecks.io/)
[![Uptime Kuma](docs/static/uptimekuma.png)](https://uptime.kuma.pet/)
[![Cronitor](docs/static/cronitor.png)](https://cronitor.io/)
[![Cronhub](docs/static/cronhub.png)](https://cronhub.io/)
[![PagerDuty](docs/static/pagerduty.png)](https://www.pagerduty.com/)
[![Pushover](docs/static/pushover.png)](https://www.pushover.net/)
[![ntfy](docs/static/ntfy.png)](https://ntfy.sh/)
[![Loki](docs/static/loki.png)](https://grafana.com/oss/loki/)
[![Apprise](docs/static/apprise.png)](https://github.com/caronc/apprise/wiki)
[![Zabbix](docs/static/zabbix.png)](https://www.zabbix.com/)
[![Sentry](docs/static/sentry.png)](https://sentry.io/)

### Credentials

[![systemd](docs/static/systemd.png)](https://systemd.io/)
[![Docker](docs/static/docker.png)](https://www.docker.com/)
[![Podman](docs/static/podman.png)](https://podman.io/)
[![KeePassXC](docs/static/keepassxc.png)](https://keepassxc.org/)

## Getting Started

To get started with borgmatic, begin by [installing and configuring it](https://torsion.org/borgmatic/docs/how-to/set-up-backups/). Explore the [borgmatic documentation](https://torsion.org/borgmatic/#documentation) for detailed guides and references.

## Hosting Providers

Find reliable hosting solutions that provide direct support for Borg and borgmatic:

*   [BorgBase](https://www.borgbase.com/?utm_source=borgmatic): A Borg hosting service that features monitoring, 2FA, and append-only repositories.
*   [Hetzner](https://hetzner.cloud/?ref=v9dOJ98Ic9I8): A storage box that offers support for Borg.

## Support and Contributing

### Issues

Report issues or suggest improvements in the [issue tracker](https://projects.torsion.org/borgmatic-collective/borgmatic/issues). You'll need to [register](https://projects.torsion.org/user/sign_up?invite_code=borgmatic) or [log in](https://projects.torsion.org/user/login) to create new issues or add comments.

See the [security policy](https://torsion.org/borgmatic/docs/security-policy/) for security-related issues.

### Social

*   Follow borgmatic on [Mastodon](https://floss.social/@borgmatic).

### Chat

*   Join the `#borgmatic` IRC channel on Libera Chat via [web chat](https://web.libera.chat/#borgmatic) or a native IRC client ([ircs://irc.libera.chat:6697](ircs://irc.libera.chat:6697)).

### Other

*   For other questions or comments, contact [witten@torsion.org](mailto:witten@torsion.org).

### Contributing

The [borgmatic source code](https://projects.torsion.org/borgmatic-collective/borgmatic) is available. Also, you can access a read-only mirror on [GitHub](https://github.com/borgmatic-collective/borgmatic).

Contribute by submitting a [pull request](https://projects.torsion.org/borgmatic-collective/borgmatic/pulls) or open an [issue](https://projects.torsion.org/borgmatic-collective/borgmatic/issues). You'll need to [register](https://projects.torsion.org/user/sign_up?invite_code=borgmatic). Refer to the [borgmatic development how-to](https://torsion.org/borgmatic/docs/how-to/develop-on-borgmatic/) for details.

borgmatic is licensed under the GNU General Public License version 3 or later.

### Recent Contributors

A huge thank you to all borgmatic contributors!

{% include borgmatic/contributors.html %}
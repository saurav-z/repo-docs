# borgmatic: Secure, Configurable Backup Software

**Protect your valuable data with borgmatic, the configuration-driven backup solution that offers client-side encryption and database support.**  This README provides an overview of borgmatic, its key features, integrations, and how to get started.  Find the original repository [here](https://github.com/borgmatic-collective/borgmatic).

## Key Features:

*   **Client-Side Encryption:** Ensures your backups are secure.
*   **Configuration-Driven:** Easily configure backups via YAML files.
*   **Database Backup Support:**  Includes support for popular databases.
*   **Flexible Repository Options:** Backup to local or remote repositories.
*   **Retention Policies:** Define how long to keep your backups.
*   **Backup Validation:** Built-in checks to verify backup integrity.
*   **Custom Scripts:** Run scripts before or after backup actions.
*   **Third-Party Integrations:**  Monitor backups with various services.

## Integrations

borgmatic seamlessly integrates with a variety of services to enhance your backup experience:

### Data

*   PostgreSQL
*   MySQL
*   MariaDB
*   MongoDB
*   SQLite
*   OpenZFS
*   Btrfs
*   LVM
*   rclone
*   BorgBase

### Monitoring

*   Healthchecks
*   Uptime Kuma
*   Cronitor
*   Cronhub
*   PagerDuty
*   Pushover
*   ntfy
*   Loki
*   Apprise
*   Zabbix
*   Sentry

### Credentials

*   systemd
*   Docker
*   Podman
*   KeePassXC

## Getting Started

1.  **Installation and Configuration:**  Start by [installing and configuring borgmatic](https://torsion.org/borgmatic/docs/how-to/set-up-backups/).
2.  **Documentation:** Explore the [borgmatic how-to and reference guides](https://torsion.org/borgmatic/#documentation) for detailed information.

## Hosting Providers

For off-site backup storage, consider these Borg/borgmatic-friendly hosting providers:

*   [BorgBase](https://www.borgbase.com/?utm_source=borgmatic): Borg hosting service with support for monitoring, 2FA, and append-only repos
*   [Hetzner](https://hetzner.cloud/?ref=v9dOJ98Ic9I8): A "storage box" that includes support for Borg
*   rsync.net (compatible storage offering, but does not fund borgmatic development or hosting.)

## Support and Contributing

### Issues

Report issues or suggest feature enhancements via the [issue tracker](https://projects.torsion.org/borgmatic-collective/borgmatic/issues) (registration required).

### Social

*   Follow borgmatic on Mastodon: <a rel="me" href="https://floss.social/@borgmatic">borgmatic on Mastodon</a>.

### Chat

*   Join the `#borgmatic` IRC channel on Libera Chat via <a href="https://web.libera.chat/#borgmatic">web chat</a> or a native <a href="ircs://irc.libera.chat:6697">IRC client</a>.

### Other

*   For other questions or comments, contact [witten@torsion.org](mailto:witten@torsion.org).

### Contributing

The [source code](https://projects.torsion.org/borgmatic-collective/borgmatic) is available.  A read-only mirror is also available on [GitHub](https://github.com/borgmatic-collective/borgmatic).  Contribute by submitting a [pull request](https://projects.torsion.org/borgmatic-collective/borgmatic/pulls) or opening an [issue](https://projects.torsion.org/borgmatic-collective/borgmatic/issues) (registration required).

Check out the [borgmatic development how-to](https://torsion.org/borgmatic/docs/how-to/develop-on-borgmatic/) for development details.

### Recent contributors

Thanks to all borgmatic contributors! There are multiple ways to contribute to
this project, so the following includes those who have fixed bugs, contributed
features, *or* filed tickets.

{% include borgmatic/contributors.html %}
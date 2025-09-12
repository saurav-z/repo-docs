# Download and Manage Your eBooks with Ease: Calibre-Web Automated Book Downloader

Effortlessly search, download, and organize your ebooks with the Calibre-Web Automated Book Downloader, a user-friendly interface designed to integrate seamlessly with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated-book-downloader) (original repository).

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder.
*   🔌 **Seamless Integration:** Works directly with Calibre-Web-Automated for streamlined library management.
*   📖 **Multi-Format Support:** Supports common ebook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capability for reliable downloads.
*   🐳 **Dockerized Deployment:** Simple, containerized setup with Docker.
*   🧅 **Tor Variant:** Anonymize your traffic using the Tor network.
*   ✅ **External Cloudflare Resolver:** Integrate with external services for enhanced Cloudflare bypass.

## Screenshots

*(Screenshots included in original readme)*

## Quick Start: Get Up and Running

### Prerequisites:

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation:

1.  Get the `docker-compose.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  Start the service:

    ```bash
    docker compose up -d
    ```

3.  Access the web interface at `http://localhost:8084`.

## Configuration: Customize Your Experience

### Environment Variables:

Customize the behavior of the application using the following environment variables.

#### Application Settings:

| Variable          | Description             | Default Value      |
| ----------------- | ----------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port      | `8084`             |
| `FLASK_HOST`      | Web interface binding   | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle       | `false`            |
| `INGEST_DIR`      | Book download directory | `/cwa-book-ingest` |
| `TZ`              | Container timezone      | `UTC`              |
| `UID`             | Runtime user ID         | `1000`             |
| `GID`             | Runtime group ID        | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database  | None               |
| `ENABLE_LOGGING`  | Enable log file         | `true`             |
| `LOG_LEVEL`       | Log level to use        | `info`             |

*   To enable authentication, set `CWA_DB_PATH` to your Calibre-Web's `app.db` path.
*   Logging is enabled by default; logs are stored in `/var/log/cwa-book-downloader` with levels from `DEBUG` to `CRITICAL`.
*   If using TOR, the TZ will be calculated automatically based on IP.

#### Download Settings:

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                    | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                     | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                           | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                    | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                              | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                  | `false`                           |
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB first instead of AA | `false`                           |

*   Set `BOOK_LANGUAGE` to a comma separated list for multiple languages.

#### Anna's Archive (AA) Settings:

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   Donator keys for faster downloads are supported via `AA_DONATOR_KEY`.
*   If Cloudflare bypass is disabled, alternative download hosts (libgen, z-lib, etc.) will be used.

#### Network Settings:

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Configure proxies using `HTTP_PROXY` and `HTTPS_PROXY`.
*   Customize DNS settings with `CUSTOM_DNS`; supports IP addresses or preset providers (google, quad9, cloudflare, opendns).
*   Consider using `USE_DOH=true` with `CUSTOM_DNS=cloudflare` for enhanced privacy.

#### Custom Configuration:

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   Execute custom scripts post-download using `CUSTOM_SCRIPT`.
*   The script receives the downloaded file path as an argument.
*   Ensure your script preserves the original filename.
*   The file will be moved to `/cwa-book-ingest` after the script execution (if not deleted)

### Volume Configuration:

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   Ensure your local path aligns with your Calibre-Web-Automated ingest folder.
*   If using a CIFS share, add `nobrl` to your fstab mount option to prevent "database locked" errors.

## Tor Variant: Enhanced Privacy

*   Uses the Tor network for anonymized downloads.
*   Requires the `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is determined by the Tor exit node when running in Tor mode.
*   Custom DNS, DoH, and proxy settings are ignored in the Tor variant.
*   Get the `docker-compose.tor.yml` to use the Tor version.

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

## External Cloudflare Resolver Variant:

*   Leverages an external service (like FlareSolverr or ByParr) to bypass Cloudflare.
*   Improves reliability and performance, especially with a dedicated resolver infrastructure.
*   Make sure to enable  `USE_CF_BYPASS` to enable this feature
*   Get the `docker-compose.extbp.yml` file to use the External Cloudflare resolver version.

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

## Architecture:

*   Single service application: `calibre-web-automated-bookdownloader`.

## Health Monitoring:

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass service.
*   Checks run every 30 seconds with a 30-second timeout and 3 retries.

## Logging:

*   Logs are available in container `/var/logs/cwa-book-downloader.log` or via Docker logs (`docker logs`).

## Contributing:

*   Contributions are welcome via Pull Requests.

## License:

*   MIT License - see the [LICENSE](LICENSE) file for details.

## Important Disclaimers:

### Copyright Notice:

*   Users are responsible for complying with copyright laws and ensuring they have the right to download content.

### Duplicate Downloads Warning:

*   The current version does not check for existing files or Calibre database entries. Please be mindful of duplicates.

## Support:

*   Report issues and ask questions by opening an issue on the GitHub repository.
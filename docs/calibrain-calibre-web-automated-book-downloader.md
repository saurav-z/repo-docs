# Automate Your eBook Downloads with Calibre-Web-Automated Book Downloader

**Tired of manually downloading eBooks?** [Calibre-Web-Automated Book Downloader](https://github.com/calibrain/calibre-web-automated-book-downloader) provides a streamlined web interface for searching and downloading books, integrating seamlessly with your Calibre library.

## Key Features:

*   **User-Friendly Web Interface:** Easily search and request books.
*   **Automated Downloads:** Books are automatically downloaded to your specified ingest folder.
*   **Seamless Integration:** Works perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) to streamline your ebook workflow.
*   **Multiple Format Support:** Download books in various formats: epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   **Cloudflare Bypass:** Built-in capability to bypass Cloudflare protection for reliable downloads.
*   **Docker Deployment:** Easy setup and management with Docker and Docker Compose.
*   **Tor Variant:** Enhanced privacy with Tor network integration.
*   **External Cloudflare Resolver Variant:** Integrated support for external Cloudflare resolvers for improved performance.

## Getting Started: Quick Setup

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation

1.  Get the `docker-compose.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  Start the service:

    ```bash
    docker compose up -d
    ```

3.  Access the web interface at `http://localhost:8084`

## Configuration Options

Customize your experience with the following environment variables:

### Application Settings

*   `FLASK_PORT`: Web interface port (default: 8084)
*   `FLASK_HOST`: Web interface binding (default: 0.0.0.0)
*   `DEBUG`: Debug mode toggle (default: false)
*   `INGEST_DIR`: Book download directory (default: `/cwa-book-ingest`)
*   `TZ`: Container timezone (default: UTC)
*   `UID`: Runtime user ID (default: 1000)
*   `GID`: Runtime group ID (default: 100)
*   `CWA_DB_PATH`: Path to Calibre-Web's database to enable authentication.  (Example: `/auth/app.db:ro`)
*   `ENABLE_LOGGING`: Enable log file (default: true), Logs are located at `/var/log/cwa-book-downloader`
*   `LOG_LEVEL`: Log level to use (default: info). Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Download Settings

*   `MAX_RETRY`: Maximum retry attempts (default: 3)
*   `DEFAULT_SLEEP`: Retry delay (seconds) (default: 5)
*   `MAIN_LOOP_SLEEP_TIME`: Processing loop delay (seconds) (default: 5)
*   `SUPPORTED_FORMATS`: Supported book formats (default: `epub,mobi,azw3,fb2,djvu,cbz,cbr`)
*   `BOOK_LANGUAGE`: Preferred language for books (default: `en`), accepts comma separated values
*   `AA_DONATOR_KEY`: Optional Donator key for Anna's Archive fast download API
*   `USE_BOOK_TITLE`: Use book title as filename instead of ID (default: false)
*   `PRIORITIZE_WELIB`: When downloading, download from WELIB first instead of AA (default: false)

### Anna's Archive (AA) Settings

*   `AA_BASE_URL`: Base URL of Annas-Archive (default: `https://annas-archive.org`)
*   `USE_CF_BYPASS`: Disable CF bypass and use alternative links instead (default: `true`)

### Network Settings

*   `AA_ADDITIONAL_URLS`: Proxy URLs for AA (, separated)
*   `HTTP_PROXY`: HTTP proxy URL
*   `HTTPS_PROXY`: HTTPS proxy URL
*   `CUSTOM_DNS`: Custom DNS IP.  Use a comma separated list for multiple DNS servers.  Can also use preset DNS providers: `google`, `quad9`, `cloudflare`, `opendns`.
*   `USE_DOH`: Use DNS over HTTPS (default: false), requires custom DNS to be set.

### Custom Configuration

*   `CUSTOM_SCRIPT`: Path to an executable script that runs after each download.  (Example: `/scripts/process-book.sh`)

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** If your library volume is on a cifs share, add `nobrl` to your mount line in your fstab file.

## Variants

### 🧅 Tor Variant

Enhance privacy by routing all traffic through the Tor network.

1.  Get the Tor-specific `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

    *Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.*

### 🌐 External Cloudflare Resolver Variant

Utilize an external service to bypass Cloudflare protection.

1.  Get the extbp-specific `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

    *Requires setting `EXT_BYPASSER_URL`.*

## Architecture

*   **calibre-web-automated-bookdownloader:** Main application, providing the web interface and download functionality.

## Health Monitoring

*   Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection. Checks run every 30 seconds with a 30-second timeout and 3 retries.

## Logging

*   Logs are available in the container at `/var/logs/cwa-book-downloader.log` and via Docker logs.

## Contributing

Contributions are welcome!  Please submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only. Users are responsible for ensuring they have the right to download requested materials and for using the tool in compliance with all applicable laws and regulations.

### Duplicate Downloads Warning

The current version *does not* check for existing files in the download directory or in your Calibre database.

## Support

Please file an issue on the GitHub repository for any questions or issues.
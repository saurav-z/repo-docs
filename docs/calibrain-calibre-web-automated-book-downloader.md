# Automate Your eBook Library: Calibre-Web Automated Book Downloader

**Effortlessly search, download, and integrate eBooks into your Calibre library with this intuitive web interface. [View the GitHub Repository](https://github.com/calibrain/calibre-web-automated-book-downloader)**

This project simplifies the process of finding and organizing eBooks, integrating seamlessly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).

## Key Features:

*   🌐 **User-Friendly Web Interface:** Easy-to-use interface for searching and requesting book downloads.
*   🔄 **Automated Downloads:**  Downloads books directly to your specified ingest folder.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated.
*   📖 **Multiple Format Support:**  Supports popular formats like epub, mobi, azw3, and more.
*   🛡️ **Cloudflare Bypass:**  Includes a Cloudflare bypass capability for reliable downloads.
*   🐳 **Dockerized Deployment:**  Easy to set up with Docker for quick deployment and portability.
*   🧅 **Tor Variant:**  Includes a Tor variant for enhanced privacy and bypassing network restrictions.
*   🌐 **External Cloudflare Resolver Variant:**  Integrates with external Cloudflare resolvers for advanced bypass capabilities.

## Getting Started

### Prerequisites:

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation:

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the Web Interface:** Navigate to `http://localhost:8084` in your web browser.

## Configuration

### Environment Variables:

This application is highly configurable. Below are the available environment variables:

#### Application Settings

*   `FLASK_PORT` - Web interface port (`8084`)
*   `FLASK_HOST` - Web interface binding (`0.0.0.0`)
*   `DEBUG` - Debug mode toggle (`false`)
*   `INGEST_DIR` - Book download directory (`/cwa-book-ingest`)
*   `TZ` - Container timezone (`UTC`)
*   `UID` - Runtime user ID (`1000`)
*   `GID` - Runtime group ID (`100`)
*   `CWA_DB_PATH` - Calibre-Web's database (Requires Authentication, points to `app.db`)
*   `ENABLE_LOGGING` - Enable log file (`true`)
*   `LOG_LEVEL` - Log level to use (`info`)

#### Download Settings

*   `MAX_RETRY` - Maximum retry attempts (`3`)
*   `DEFAULT_SLEEP` - Retry delay (seconds) (`5`)
*   `MAIN_LOOP_SLEEP_TIME` - Processing loop delay (seconds) (`5`)
*   `SUPPORTED_FORMATS` - Supported book formats (`epub,mobi,azw3,fb2,djvu,cbz,cbr`)
*   `BOOK_LANGUAGE` - Preferred language for books (`en`). Accepts multiple comma-separated languages like `en,fr,ru`.
*   `AA_DONATOR_KEY` - Optional Donator key for Anna's Archive fast download API
*   `USE_BOOK_TITLE` - Use book title as filename instead of ID (`false`)
*   `PRIORITIZE_WELIB` - When downloading, download from WELIB first instead of AA (`false`)

#### Anna's Archive Settings (`AA`)

*   `AA_BASE_URL` - Base URL of Annas-Archive (can be changed for a proxy) (`https://annas-archive.org`)
*   `USE_CF_BYPASS` - Disable CF bypass and use alternative links instead (`true`)

#### Network Settings

*   `AA_ADDITIONAL_URLS` - Proxy URLs for AA (, separated) (``)
*   `HTTP_PROXY` - HTTP proxy URL (``)
*   `HTTPS_PROXY` - HTTPS proxy URL (``)
*   `CUSTOM_DNS` - Custom DNS IP (``)
*   `USE_DOH` - Use DNS over HTTPS (`false`)

#### Custom Configuration

*   `CUSTOM_SCRIPT` - Path to an executable script that runs after each download (``)

### Volume Configuration

Mount your ingest folder to the volume:

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
```

### Tor Variant

*   This variant routes all traffic through the Tor network for enhanced privacy.
*   Uses the `docker-compose.tor.yml` file.
*   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.

### External Cloudflare Resolver Variant

*   Uses an external service (like FlareSolverr) to bypass Cloudflare.
*   Uses the `docker-compose.extbp.yml` file.
*   Enables the Cloudflare bypass through `USE_CF_BYPASS`.
*   Requires `EXT_BYPASSER_URL`.
*   Requires `EXT_BYPASSER_PATH`.
*   Requires `EXT_BYPASSER_TIMEOUT`.

## Architecture:

The application consists of one core service:

1.  **calibre-web-automated-bookdownloader**:  The main application, providing the web interface and download functionality.

## Health Monitoring:

Built-in health checks monitor:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

Checks run every 30 seconds with a 30-second timeout and 3 retries.
You can enable by adding this to your compose :
```
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pyrequests http://localhost:8084/request/api/status || exit 1
```

## Logging:

Logs can be found in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Access via `docker logs`

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Copyright Notice

This tool accesses sources that may contain copyrighted material. **Users are solely responsible for ensuring they have the right to download and use requested materials and for complying with copyright laws.**

### Duplicate Downloads

This tool does not currently check for existing files or verify if books already exist in your Calibre database.
**Exercise caution when requesting multiple books to avoid duplicates.**

## Support

For issues or questions, please open an issue on the GitHub repository.
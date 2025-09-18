# 📚 Automate Your eBook Library: Calibre-Web Automated Book Downloader

Streamline your eBook workflow with the **Calibre-Web Automated Book Downloader**, a user-friendly web interface for searching, downloading, and seamlessly integrating books into your Calibre library.  [Check out the original repository](https://github.com/calibrain/calibre-web-automated-book-downloader).

## ✨ Key Features

*   **Intuitive Web Interface:** Easily search and request book downloads.
*   **Automated Downloads:**  Books are automatically downloaded to your designated ingest folder.
*   **Seamless Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   **Multi-Format Support:**  Download books in various formats including epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   **Cloudflare Bypass:** Built-in capability to reliably download books.
*   **Docker Deployment:** Easy setup and management using Docker.
*   **Tor & External Cloudflare Resolver Variants**: Enhanced privacy and resolver flexibility.

## 🚀 Quick Start

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

## ⚙️ Configuration

### Environment Variables

#### Application Settings

*   `FLASK_PORT`: Web interface port (Default: `8084`)
*   `FLASK_HOST`: Web interface binding (Default: `0.0.0.0`)
*   `DEBUG`: Debug mode toggle (Default: `false`)
*   `INGEST_DIR`: Book download directory (Default: `/cwa-book-ingest`)
*   `TZ`: Container timezone (Default: `UTC`)
*   `UID`: Runtime user ID (Default: `1000`)
*   `GID`: Runtime group ID (Default: `100`)
*   `CWA_DB_PATH`: Path to Calibre-Web's database (`app.db`) - required for authentication if enabled
*   `ENABLE_LOGGING`: Enable log file (Default: `true`)
*   `LOG_LEVEL`: Log level to use (Default: `info`) - Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    *   If logging is enabled, the default log folder is `/var/log/cwa-book-downloader`.
    *   When using TOR, the TZ is automatically calculated based on IP.

#### Download Settings

*   `MAX_RETRY`: Maximum retry attempts (Default: `3`)
*   `DEFAULT_SLEEP`: Retry delay in seconds (Default: `5`)
*   `MAIN_LOOP_SLEEP_TIME`: Processing loop delay in seconds (Default: `5`)
*   `SUPPORTED_FORMATS`: Supported book formats (Default: `epub,mobi,azw3,fb2,djvu,cbz,cbr`)
*   `BOOK_LANGUAGE`: Preferred language for books (Default: `en`) - Supports comma-separated values (e.g., `en,fr,ru`).
*   `AA_DONATOR_KEY`: Optional Donator key for Anna's Archive fast download API.
*   `USE_BOOK_TITLE`: Use book title as filename instead of ID (Default: `false`)
*   `PRIORITIZE_WELIB`: Download from WELIB first instead of AA (Default: `false`)

#### AA Settings

*   `AA_BASE_URL`: Base URL of Annas-Archive (Default: `https://annas-archive.org`)
*   `USE_CF_BYPASS`: Disable CF bypass and use alternative links instead (Default: `true`)

#### Network Settings

*   `AA_ADDITIONAL_URLS`: Proxy URLs for AA (comma separated) (Default: ``)
*   `HTTP_PROXY`: HTTP proxy URL (Default: ``)
*   `HTTPS_PROXY`: HTTPS proxy URL (Default: ``)
*   `CUSTOM_DNS`: Custom DNS IP or Preset DNS Provider (Default: ``)  - Supports IPv4/IPv6 and preset providers: google, quad9, cloudflare, opendns.
*   `USE_DOH`: Use DNS over HTTPS (Default: `false`)

    *   To utilize DoH, set a `CUSTOM_DNS` provider (google, quad9, cloudflare, opendns) and set `USE_DOH=true`.

#### Custom Configuration

*   `CUSTOM_SCRIPT`: Path to an executable script that runs after each download (Default: ``)

    *   The script receives the full path of the downloaded file as an argument.
    *   The script should retain the original filename and can be modified, or deleted. The file is moved to `/cwa-book-ingest` after the script execution (if not deleted).

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** If using a cifs share for your library volume, add `nobrl` to the mount line in your `/etc/fstab` file.

## 🧅 Tor Variant

This variant uses the Tor network for enhanced privacy.

1.  Get the Tor-specific `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Considerations for Tor:**

*   This requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is determined by the Tor exit node's IP.
*   Network settings such as custom DNS or proxies are ignored.

## 🚀 External Cloudflare Resolver Variant

Utilize an external service (like FlareSolverr) to bypass Cloudflare protection.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

#### Instructions:

1.  Get the external Cloudflare resolver `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   Enable `USE_CF_BYPASS` to use this variant.
*   This feature is compatible with resolvers that implement the `FlareSolverr` API.

## 🏗️ Architecture

The application comprises a single service: `calibre-web-automated-bookdownloader`.

## 🏥 Health Monitoring

Built-in health checks monitor:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

The checks run every 30 seconds with a 30-second timeout and 3 retries.

## 📝 Logging

Logs can be found in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome; submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool may access sources with copyrighted material. Users are responsible for ensuring they have the right to download requested materials and must respect copyright laws and intellectual property rights.

### Duplicate Downloads

This application does not check for existing files or verify book presence in your Calibre database, so be mindful of potential duplicate downloads.
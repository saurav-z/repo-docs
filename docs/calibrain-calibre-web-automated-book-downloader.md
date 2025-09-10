# 📚 Automate Your eBook Library with Calibre-Web-Automated Book Downloader

**Effortlessly search, download, and manage your eBooks with a user-friendly interface designed to integrate seamlessly with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated-book-downloader).**

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Easily search for and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder, ready for Calibre integration.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated for streamlined library management.
*   📖 **Wide Format Support:**  Download books in various formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:**  Includes Cloudflare bypass capabilities for reliable downloads.
*   🐳 **Dockerized Deployment:**  Simple Docker-based deployment for quick setup and easy management.
*   🧅 **Tor Variant:** Optional Tor integration for enhanced privacy and network flexibility.
*   🌐 **External Cloudflare Resolver:** Utilize external services for Cloudflare bypass.

## 🖼️ Screenshots

*   ![Main search interface Screenshot](README_images/search.png 'Main search interface')
*   ![Details modal Screenshot placeholder](README_images/details.png 'Details modal')
*   ![Download queue Screenshot placeholder](README_images/downloading.png 'Download queue')

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

1.  Get the `docker-compose.yml`:

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

*   **Authentication:** To enable authentication, set `CWA_DB_PATH` to point to Calibre-Web's `app.db`.
*   **Logging:**  If enabled, logs are located in `/var/log/cwa-book-downloader`. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Timezone:**  When using Tor, the timezone is automatically set based on the Tor exit node's IP.

#### Download Settings

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

*   **Multiple Languages:**  Set `BOOK_LANGUAGE` to a comma-separated list (e.g., `en,fr,ru`).

#### AA (Anna's Archive) Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   **AA Donator Key:**  Use your key in `AA_DONATOR_KEY` for faster downloads (if you are a donor).
*   **Cloudflare Bypass:**  If disabling the bypass, alternative download hosts like Libgen or Z-Lib are used (may have delays).

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:**

    ```bash
    HTTP_PROXY=http://proxy.example.com:8080
    HTTPS_PROXY=http://proxy.example.com:8080
    HTTP_PROXY=http://username:password@proxy.example.com:8080
    HTTPS_PROXY=http://username:password@proxy.example.com:8080
    ```
*   **Custom DNS:**

    1.  Comma-separated IP addresses (e.g., `127.0.0.53,127.0.1.53`)
    2.  Preset providers: `google`, `quad9`, `cloudflare`, `opendns`
*   **DNS over HTTPS (DoH):**  Enable with `USE_DOH=true` when using a preset DNS provider.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Custom Script:**  Set `CUSTOM_SCRIPT` to execute a script after each successful download for post-processing (e.g., format conversion).
    *   The script receives the full path of the downloaded file as an argument.
    *   The script must preserve the original filename.
    *   The file will be moved to `/cwa-book-ingest` after script execution.

    Example:

    ```yaml
    environment:
      - CUSTOM_SCRIPT=/scripts/process-book.sh
    volumes:
      - local/scripts/custom_script.sh:/scripts/process-book.sh
    ```

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** If your library volume is on a CIFS share, add `nobrl` to your mount line in `fstab` to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

Utilize the Tor network for enhanced privacy and bypassing network restrictions.

1.  Get the Tor-specific `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start the service:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Tor Considerations:**  Requires `NET_ADMIN` and `NET_RAW` Docker capabilities. Timezone is determined by the Tor exit node.  Custom DNS, DoH, and proxy settings are ignored.

### External Cloudflare Resolver Variant

Leverage an external service to bypass Cloudflare protection, providing greater reliability.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  Get the extbp-specific docker-compose file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   **Compatibility:** Works with resolvers implementing the `FlareSolverr` API schema (e.g., ByParr).
*   **Enable `USE_CF_BYPASS` configuration**.

## 🏗️ Architecture

*   **calibre-web-automated-bookdownloader:** The core application providing the web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor:
    *   Web interface availability
    *   Download service status
    *   Cloudflare bypass service connection
*   Checks run every 30 seconds with a 30-second timeout and 3 retries.  Can be enabled through docker-compose
    ```
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
        CMD pyrequests http://localhost:8084/request/api/status || exit 1
    ```

## 📝 Logging

*   Logs are available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome! Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only. Users are responsible for complying with copyright laws and intellectual property rights.

### Duplicate Downloads Warning

The current version does not:

*   Check for existing files in the download directory
*   Verify if books already exist in your Calibre database

## 💬 Support

For issues or questions, please file an issue on the GitHub repository at [https://github.com/calibrain/calibre-web-automated-book-downloader](https://github.com/calibrain/calibre-web-automated-book-downloader).
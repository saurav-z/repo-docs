# 📚 Automate Your eBook Downloads with the Calibre-Web Book Downloader

**Tired of manually downloading eBooks?** This user-friendly web interface seamlessly integrates with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) to simplify searching, requesting, and downloading books directly to your library.

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Search, request, and download books with ease.
*   🔄 **Automated Downloads:** Downloads directly to your Calibre-Web-Automated ingest directory.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated for a streamlined workflow.
*   📖 **Multi-Format Support:** Supports popular formats like EPUB, MOBI, AZW3, and more.
*   🛡️ **Cloudflare Bypass:** Reliably downloads books even with Cloudflare protection.
*   🐳 **Docker-Based Deployment:** Quick and easy setup with Docker Compose.

## 🖼️ Screenshots

*   ![Main Search Interface](README_images/search.png 'Main Search Interface')
*   ![Details Modal](README_images/details.png 'Details Modal')
*   ![Download Queue](README_images/downloading.png 'Download Queue')

## 🚀 Getting Started

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:** Navigate to `http://localhost:8084` in your web browser.

## ⚙️ Configuration

Customize the application's behavior with environment variables:

### Application Settings

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

*   **Authentication:** Enable by setting `CWA_DB_PATH` to your Calibre-Web `app.db` file.
*   **Logging:** Enabled by default, logs to `/var/log/cwa-book-downloader`.  Available log levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Timezone:** Automatically set when using the Tor variant.

### Download Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                    | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                     | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                           | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                    | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                              | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                  | `false`                           |
| `PRIORITIZE_WELIB`     | Download from WELIB first instead of AA                    | `false`                           |

*   **Book Language:** Supports multiple languages (e.g., `en,fr,ru`).

### Anna's Archive Settings

| Variable               | Description                                                 | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Anna's Archive Base URL  (Can be changed to use a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable Cloudflare bypass and use alternative links instead       | `true`                            |

*   **AA Donator Key:**  Use your key for faster downloads from Anna's Archive.
*   **Cloudflare Bypass:** If disabled uses alternative download hosts like libgen or z-lib.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:** Supports HTTP and HTTPS proxies with or without authentication.
*   **Custom DNS:** Use custom DNS servers or preset providers (Google, Quad9, Cloudflare, OpenDNS).
*   **DNS over HTTPS (DoH):** Can be enabled when using a custom DNS provider.

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Post-Download Script:** Execute a custom script after each successful download.  The script receives the file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Important:** If your library volume is on a CIFS share, add `nobrl` to your `fstab` mount options to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

For enhanced privacy and to bypass network restrictions, utilize the Tor variant.

1.  **Get the Tor Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Tor Considerations:**

*   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is set automatically based on the Tor exit node's IP.
*   Custom DNS, DoH, and proxy settings are ignored.

### External Cloudflare Resolver Variant

Leverage an external service (like FlareSolverr) for more reliable Cloudflare bypass.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

#### Getting Started

1.  **Get the extbp Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Benefits:

- Centralizes Cloudflare bypass logic for easier maintenance.
- Can leverage more powerful or distributed resolver infrastructure.
- Reduces load on the main application container.

## 🏗️ Architecture

*   Consists of a single service: `calibre-web-automated-bookdownloader`

## 🏥 Health Monitoring

*   Built-in health checks every 30 seconds (30-second timeout, 3 retries).

## 📝 Logging

*   Logs available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome!  Submit a pull request.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

**This tool is for legitimate use only.** Users are responsible for respecting copyright laws and intellectual property rights.

### Duplicate Downloads

The current version **does not** check for existing files or verify books in your Calibre database. Exercise caution to avoid duplicates.

## 💬 Support

For issues or questions, please file an issue on the GitHub repository.
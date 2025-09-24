# Automate Your Ebook Library: Calibre-Web Automated Book Downloader

**Effortlessly search, download, and manage your ebooks with a user-friendly interface designed for seamless integration with Calibre-Web-Automated. ([Original Repo](https://github.com/calibrain/calibre-web-automated-book-downloader))**

## 📚 Key Features

*   🌐 **Intuitive Web Interface:** Easily search for and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work flawlessly with Calibre-Web-Automated.
*   📖 **Multiple Format Support:** Supports common ebook formats like EPUB, MOBI, AZW3, FB2, DJVU, CBZ, and CBR.
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass functionality for reliable downloads.
*   🐳 **Docker Deployment:** Simple, containerized setup for quick and easy installation.

## 🖼️ Screenshots

[Placeholder for screenshots of the web interface, detail modal and download queue]

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

3.  **Access the web interface:** Open your web browser and go to `http://localhost:8084`

## ⚙️ Configuration

Configure your application's behavior using environment variables.

### Application Settings

| Variable          | Description                       | Default Value      |
| ----------------- | --------------------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port                | `8084`             |
| `FLASK_HOST`      | Web interface binding             | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle                 | `false`            |
| `INGEST_DIR`      | Book download directory           | `/cwa-book-ingest` |
| `TZ`              | Container timezone                | `UTC`              |
| `UID`             | Runtime user ID                   | `1000`             |
| `GID`             | Runtime group ID                  | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database path      | None               |
| `ENABLE_LOGGING`  | Enable log file                   | `true`             |
| `LOG_LEVEL`       | Log level to use (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `info`             |

*   To enable authentication, set `CWA_DB_PATH` to match your Calibre-Web's `app.db`.
*   Logs are stored in `/var/log/cwa-book-downloader` when logging is enabled.

### Download Settings

| Variable               | Description                                                       | Default Value                     |
| ---------------------- | ----------------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum download retry attempts                                    | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                             | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                                     | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                            | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books (comma-separated)                    | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API           | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                        | `false`                           |
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB first instead of AA      | `false`                           |

### AA (Anna's Archive) Settings

| Variable               | Description                                                                 | Default Value                     |
| ---------------------- | ----------------------------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)                       | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable Cloudflare bypass and use alternative links instead                   | `true`                            |

*   Use your Anna's Archive Donator key in `AA_DONATOR_KEY` for faster downloads.
*   Disabling `USE_CF_BYPASS` uses alternative download hosts if available.

### Network Settings

| Variable               | Description                                           | Default Value           |
| ---------------------- | ----------------------------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for Anna's Archive (comma separated)      | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                                        | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                                       | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP or preset providers (e.g., cloudflare) | ``                      |
| `USE_DOH`              | Use DNS over HTTPS                                  | `false`                 |

*   Configure proxies using the format: `HTTP_PROXY=http://username:password@proxy.example.com:8080`
*   Use custom DNS servers or preset providers to bypass network restrictions (e.g., ISP blocks).
*   Set `USE_DOH=true` with a supported `CUSTOM_DNS` provider for DNS over HTTPS.

### Custom Configuration

| Variable               | Description                                                                   | Default Value           |
| ---------------------- | ----------------------------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that runs after each download                    | ``                      |

*   Use a custom script (`CUSTOM_SCRIPT`) for post-download processing, e.g., format conversion or validation.
*   The script receives the full downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Important:** If using a CIFS share for your library, add `nobrl` to your `fstab` mount options to prevent "database locked" errors.

## 🧅 Tor Variant

This variant routes all traffic through the Tor network for enhanced privacy.

1.  **Get the Tor-specific `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  **Start the service using the Tor compose file:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```
    *   **Important Considerations for Tor:** Requires Docker capabilities for transparent Tor proxying. Custom DNS, DoH, and proxy settings are ignored.  Timezone is calculated from the Tor exit node's IP.

## 🌐 External Cloudflare Resolver Variant

This variant utilizes an external Cloudflare resolver for improved bypass performance and reliability.

#### Configuration

| Variable               | Description                                           | Default Value           |
| ---------------------- | ----------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)     |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)            | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)           | `60000`                 |

*   **Use the `USE_CF_BYPASS` to enable.**
*   Compatible with resolvers that implement the FlareSolverr API (e.g., FlareSolverr, ByParr)

1.  **Get the extbp-specific `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  **Start the service using the extbp compose file:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

## 🏗️ Architecture

*   **calibre-web-automated-bookdownloader:** The main application with the web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass service.
*   Checks run every 30 seconds with a 30-second timeout and 3 retries.

## 📝 Logging

*   Logs available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

We welcome contributions! Submit Pull Requests via our GitHub repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only. Users are responsible for ensuring they have the rights to download and use requested materials, complying with all copyright laws.

### Duplicate Downloads

The current version does not check for existing files or books in your Calibre database. Be cautious to avoid duplicates.

## 💬 Support

For assistance, please file an issue on the GitHub repository.
# 📚 Effortlessly Download Books for Your Calibre Library with Calibre-Web-Automated Book Downloader

Tired of manually searching and downloading books for your Calibre library?  **Calibre-Web-Automated Book Downloader** streamlines your book acquisition process with an intuitive web interface and seamless integration with Calibre-Web-Automated.  [Check out the original repo](https://github.com/calibrain/calibre-web-automated-book-downloader).

## ✨ Key Features:

*   🌐 **User-Friendly Web Interface:** Easily search and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your designated ingest folder.
*   🔌 **Seamless Integration:** Designed to work flawlessly with Calibre-Web-Automated.
*   📖 **Multiple Format Support:** Supports common ebook formats like epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   🛡️ **Cloudflare Bypass:** Built-in capability for reliable downloads, bypassing Cloudflare protection.
*   🐳 **Docker Deployment:** Easy setup and management with Docker.
*   🧅 **Tor Variant:** Access the service securely and privately via the Tor network.
*   ☁️ **External Cloudflare Resolver Integration:** Use an external service like FlareSolverr for advanced Cloudflare bypass (more reliable).

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

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

Configure the behavior of the application with environment variables:

### Application Settings

| Variable          | Description                     | Default Value      |
| ----------------- | ------------------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port              | `8084`             |
| `FLASK_HOST`      | Web interface binding           | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle               | `false`            |
| `INGEST_DIR`      | Book download directory       | `/cwa-book-ingest` |
| `TZ`              | Container timezone              | `UTC`              |
| `UID`             | Runtime user ID                 | `1000`             |
| `GID`             | Runtime group ID                | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database          | None               |
| `ENABLE_LOGGING`  | Enable log file                 | `true`             |
| `LOG_LEVEL`       | Log level to use                | `info`             |

*   If you enable authentication, you must set `CWA_DB_PATH` to point to Calibre-Web's `app.db`.
*   Logs are saved to `/var/log/cwa-book-downloader` when logging is enabled.

### Download Settings

| Variable               | Description                                         | Default Value                     |
| ---------------------- | --------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                              | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                               | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                     | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                              | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                        | `en`                              |
| `AA_DONATOR_KEY`       | Anna's Archive Donator key (for faster downloads) | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID            | `false`                           |
| `PRIORITIZE_WELIB`     | Download from WELIB before Anna's Archive          | `false`                           |

*   Set `BOOK_LANGUAGE` to a comma-separated list (e.g., `en,fr,ru`) for multiple language preferences.

### Anna's Archive (AA) Settings

| Variable               | Description                                                   | Default Value                     |
| ---------------------- | ------------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)        | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead           | `true`                            |

*   Enter your Donator Key to increase download speed.
*   When `USE_CF_BYPASS` is disabled, the application uses alternative download hosts.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Configure proxy settings using standard formats.  
*   Configure `CUSTOM_DNS` for custom DNS servers or preset providers (e.g., `cloudflare`). Consider enabling `USE_DOH=true` with a `CUSTOM_DNS` provider for enhanced privacy.

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   Execute custom scripts after each download, useful for format conversion.  The script receives the downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   Mount your book library directory.  If using a CIFS share, add `nobrl` to your fstab mount options to prevent database lock errors.

## 🧅 Tor Variant

For increased privacy, use the Tor-specific Docker Compose file:

1.  Get the `docker-compose.tor.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   The Tor variant requires `NET_ADMIN` and `NET_RAW` Docker capabilities and will determine the timezone automatically. DNS and proxy settings will be ignored.

## ☁️ External Cloudflare Resolver Variant

Use an external Cloudflare resolver for enhanced reliability:

1.  Get the `docker-compose.extbp.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*  Set the environment variables `EXT_BYPASSER_URL`, `EXT_BYPASSER_PATH`, and `EXT_BYPASSER_TIMEOUT`.

## 🏗️ Architecture

*   A single service:  `calibre-web-automated-bookdownloader` for web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass every 30 seconds.

## 📝 Logging

*   Logs are available in the container (`/var/logs/cwa-book-downloader.log`) and via Docker logs.

## 🤝 Contributing

Contributions are welcome!  Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file.

## ⚠️ Important Disclaimers

### Copyright Notice

*   Use this tool responsibly and legally.  Users are responsible for copyright compliance.

### Duplicate Downloads Warning

*   This tool does not check for existing files or books in your Calibre library. Avoid duplicate downloads.

## 💬 Support

*   Report issues or ask questions on the GitHub repository.
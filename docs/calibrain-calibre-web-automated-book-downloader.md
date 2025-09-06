# Download Books with Ease: Calibre-Web Automated Book Downloader

**Effortlessly search, request, and download books for your Calibre library with this intuitive web interface.**  [View the project on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader).

## Key Features

*   **🌐 User-Friendly Web Interface:** A clean and intuitive interface makes finding and requesting books a breeze.
*   **🔄 Automated Downloads:**  Books are automatically downloaded to your specified ingest folder, ready for Calibre.
*   **🔌 Seamless Calibre-Web Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   **📖 Extensive Format Support:** Download books in popular formats like epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   **🛡️ Cloudflare Bypass:** Reliable downloads even with Cloudflare protection.
*   **🐳 Docker-Based Deployment:**  Easy setup and management with Docker.
*   **🧅 Tor Variant:** Option to route all traffic through the Tor network for enhanced privacy.
*   **🚀 External Cloudflare Resolver Variant:** Integrates with external Cloudflare resolvers for improved reliability.

## Getting Started

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation (Docker Compose)

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:**  `http://localhost:8084` (Adjust port if necessary)

## Configuration

Customize the application's behavior with environment variables.

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

*   **Authentication:** To enable authentication, set `CWA_DB_PATH` to point to Calibre-Web's `app.db`.
*   **Logging:** Logs are located at `/var/log/cwa-book-downloader`. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Timezone:** Automatically set in Tor mode.

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
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB first instead of AA | `false`                           |

*   **Multiple Languages:**  Set `BOOK_LANGUAGE` with comma-separated values (e.g., `en,fr,ru`).

### Anna's Archive (AA) Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   **Donator Key:**  Use your key in `AA_DONATOR_KEY` for faster downloads.
*   **Cloudflare Bypass:** Disable to use alternative download hosts.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:**  Specify `HTTP_PROXY` and `HTTPS_PROXY` with authentication if needed.
*   **Custom DNS:** Set `CUSTOM_DNS` to a comma-separated list of IP addresses or use a preset option (`google`, `quad9`, `cloudflare`, `opendns`). Consider using Cloudflare to bypass ISP blocks.  Use `USE_DOH=true` with a supported DNS provider for DNS over HTTPS.

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Custom Script:** Define a script to run after each download for custom processing, like format conversion.  The script receives the downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   **Ingest Folder:**  Mount your Calibre-Web-Automated ingest folder.
*   **CIFS Shares:**  Add `nobrl` to your fstab mount options for CIFS shares.

## Variants

### 🧅 Tor Variant

For enhanced privacy, use the Tor-specific Docker Compose file.

1.  **Get the `docker-compose.tor.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Important:**  Requires `NET_ADMIN` and `NET_RAW` Docker capabilities. Timezone is automatically set. Custom network settings are ignored.

### 🚀 External Cloudflare Resolver Variant

Integrate with external Cloudflare resolvers (e.g., FlareSolverr) for improved reliability.

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  **Get the `docker-compose.extbp.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   Requires `USE_CF_BYPASS=true`.

## Architecture

The application consists of a single service: `calibre-web-automated-bookdownloader`.

## Health Monitoring

Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection. Checks run every 30 seconds.

## Logging

Logs are available in the container at `/var/logs/cwa-book-downloader.log` and via `docker logs`.

## Contribute

Contributions are welcome! Submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

**This tool is for legitimate use only.**  Users are responsible for copyright compliance and intellectual property rights.

### Duplicate Downloads Warning

The current version *does not* check for existing files or books in your Calibre database.  Be cautious to avoid duplicates.

## Support

For issues or questions, please file an issue on the GitHub repository.
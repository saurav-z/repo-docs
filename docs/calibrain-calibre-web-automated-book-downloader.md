# 📚 Automate Your eBook Library with Calibre-Web Automated Book Downloader

**Effortlessly search, request, and download eBooks with a user-friendly interface, seamlessly integrating with your existing Calibre library.** [View the project on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader)

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Easily search and download books.
*   🔄 **Automated Downloads:** Direct downloads to your specified ingest folder.
*   🔌 **Calibre-Web Integration:** Designed for seamless compatibility.
*   📖 **Multiple Format Support:** epub, mobi, azw3, fb2, djvu, cbz, cbr, and more.
*   🛡️ **Cloudflare Bypass:** Built-in capability for reliable downloads.
*   🐳 **Docker Deployment:** Simple and quick setup.
*   🧅 **Tor Variant:** Enhanced privacy and network restriction bypassing.
*   💻 **External Cloudflare Resolver Variant:** Leverage dedicated resolver services (FlareSolverr, ByParr, etc.) for improved performance and centralized management.

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

*   **Authentication:** Set `CWA_DB_PATH` to point to Calibre-Web's `app.db` to enable authentication.
*   **Logging:** Log files are located at `/var/log/cwa-book-downloader`. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Timezone:** When using Tor, timezone is automatically set based on the exit node's IP.

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

*   **Multiple Languages:** Set `BOOK_LANGUAGE` to a comma-separated list (e.g., `en,fr,ru`).

#### AA (Anna's Archive) Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   **AA Donator Key:** Use your key in `AA_DONATOR_KEY` for faster downloads.
*   **Alternative Downloads:** Disable cloudflare bypass to use alternative download hosts (libgen, z-lib).

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:** Example: `HTTP_PROXY=http://username:password@proxy.example.com:8080`
*   **Custom DNS:** Supports IP addresses (e.g., `127.0.0.53,127.0.1.53`) or preset providers (`google`, `quad9`, `cloudflare`, `opendns`).
*   **DNS over HTTPS:** Set `USE_DOH=true` when using `CUSTOM_DNS` providers.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Custom Scripts:** Execute scripts after each successful download. The script receives the downloaded file's path as an argument. Ensure the script preserves the original filename.

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

**Note:** If your library volume is on a CIFS share, add `nobrl` to your mount options in `/etc/fstab` to avoid database locked errors.

## 🧅 Tor Variant

*   **Purpose:** Enhance privacy and bypass network restrictions using the Tor network.

1.  Get the `docker-compose.tor.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  Start the service:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Considerations:**

*   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is determined by Tor exit node's IP.
*   Custom DNS, DoH, and proxy settings are ignored.

## 💻 External Cloudflare Resolver Variant

*   **Purpose:** Integrate with external Cloudflare resolver services (e.g., FlareSolverr, ByParr) for enhanced bypass reliability and management.

*   Get the `docker-compose.extbp.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

*   Start the service:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

**Configuration:**

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

*   **Compatibility:** Designed to work with resolvers implementing the FlareSolverr API.
*   **Enable:** Set `USE_CF_BYPASS=true`

## 🏗️ Architecture

*   Consists of a single service: `calibre-web-automated-bookdownloader`.

## 🏥 Health Monitoring

*   Monitors web interface, download service, and Cloudflare bypass service.
*   Checks run every 30 seconds with a 30-second timeout and 3 retries.
*   Enable health checks in compose with:
    ```
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
        CMD pyrequests http://localhost:8084/request/api/status || exit 1
    ```

## 📝 Logging

*   Logs are available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

*   **Responsibility:** Users are responsible for ensuring they have the right to download materials, respecting copyright laws, and using the tool in compliance with local regulations.

### Duplicate Downloads Warning

*   The current version does not check for existing files or verify if books exist in your Calibre database. Exercise caution to avoid duplicates.

## 💬 Support

*   For issues or questions, please file an issue on the GitHub repository.
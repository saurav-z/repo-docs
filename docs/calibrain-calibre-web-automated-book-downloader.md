# Automate Your eBook Library: Calibre-Web Automated Book Downloader

**Streamline your eBook management with the Calibre-Web Automated Book Downloader, a user-friendly web interface that simplifies searching and downloading books for your Calibre library. ([View on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader))**

## Key Features

*   📚 **Effortless Book Search:** Intuitive web interface for easy book discovery.
*   ⬇️ **Automated Downloads:** Automatically downloads books to your designated ingest folder.
*   🔗 **Seamless Integration:** Designed to work flawlessly with Calibre-Web-Automated.
*   📖 **Wide Format Support:** Compatible with various book formats: epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capabilities for reliable downloads.
*   🐳 **Dockerized Deployment:** Simple Docker-based setup for quick and easy installation.
*   🧅 **Tor Variant:** Added support for a Tor variant for enhanced privacy.
*   ⚙️ **External Cloudflare Resolver:** Supports integration with external Cloudflare resolvers.

## Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:**  `http://localhost:8084`

## Configuration

Configure the application using environment variables:

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

*   Set `CWA_DB_PATH` to enable authentication (requires syncing with Calibre-Web's `app.db`).
*   If logging is enabled, logs are written to `/var/log/cwa-book-downloader`

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

*   Multiple languages can be specified for `BOOK_LANGUAGE` (e.g., `en,fr,ru`).

### AA (Anna's Archive) Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   Use your AA Donator Key in `AA_DONATOR_KEY` for faster downloads.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Configure proxies and custom DNS servers as needed.
*   Preset DNS providers are: `google`, `quad9`, `cloudflare`, `opendns`.
*   Use `USE_DOH=true` with a custom DNS provider for DNS over HTTPS.

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   Specify a `CUSTOM_SCRIPT` for post-download processing.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```
*   Ensure `nobrl` is added to the `cifs` mount for your shared library.

## Variants

### 🧅 Tor Variant

Use the Tor variant for enhanced privacy:

1.  **Get the Tor docker-compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

### External Cloudflare Resolver Variant

Use an external Cloudflare resolver for enhanced reliability:

1.  **Get the extbp docker-compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

*   Configure `EXT_BYPASSER_URL`, `EXT_BYPASSER_PATH`, and `EXT_BYPASSER_TIMEOUT` for your resolver.
*   Set `USE_CF_BYPASS=true` to enable the external resolver.

## Architecture

The application consists of a single service:

1.  **calibre-web-automated-bookdownloader:**  Provides the web interface and download functionality.

## Health Monitoring

Built-in health checks monitor:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

## Logging

Logs are accessible in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Use `docker logs`

## Contributing

Contributions are welcome!  Submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Copyright Notice

This tool is for legitimate use only. Users are responsible for:

*   Ensuring they have the right to download requested materials
*   Respecting copyright laws and intellectual property rights
*   Using the tool in compliance with their local regulations

### Duplicate Downloads Warning

Current version:

*   Does not check for existing files
*   Does not verify if books already exist in your Calibre database
*   Exercise caution when requesting multiple books to avoid duplicates

## Support

For support, please file an issue on the GitHub repository.
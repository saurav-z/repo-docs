# Automate Your eBook Library: Calibre-Web-Automated Book Downloader

**Effortlessly search, request, and download eBooks with a user-friendly web interface, seamlessly integrating with your Calibre library.** Visit the [original repo](https://github.com/calibrain/calibre-web-automated-book-downloader) for the latest updates.

## Key Features

*   🌐 **Intuitive Web Interface:** Easily search and request books.
*   🔄 **Automated Downloads:** Downloads directly to your specified ingest folder.
*   🔌 **Calibre-Web Integration:** Works seamlessly with Calibre-Web-Automated.
*   📖 **Multi-Format Support:** Supports common eBook formats (epub, mobi, azw3, etc.).
*   🛡️ **Cloudflare Bypass:** Includes a Cloudflare bypass for reliable downloads.
*   🐳 **Docker Deployment:** Simple, Docker-based setup for quick deployment.
*   🧅 **Tor Support:** Includes a Tor variant to anonymize your traffic and bypass network restrictions.
*   🤖 **External Cloudflare Resolver:** Integrates with services like FlareSolverr for advanced Cloudflare bypass capabilities.

## Screenshots

<details>
<summary>Click to view Screenshots</summary>

![Main search interface](README_images/search.png 'Main search interface')

![Details modal](README_images/details.png 'Details modal')

![Download queue](README_images/downloading.png 'Download queue')

</details>

## Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

1.  **Get `docker-compose.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:** `http://localhost:8084`

## Configuration

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

**Authentication:** Set `CWA_DB_PATH` to point to Calibre-Web's `app.db` to enable authentication.
**Logging:** Logs are located in `/var/log/cwa-book-downloader` if logging is enabled.  Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
**Timezone:** The timezone is automatically calculated if using TOR.

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

**`BOOK_LANGUAGE`:** You can add multiple comma separated languages, such as `en,fr,ru` etc.

#### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

**`AA_DONATOR_KEY`:** If you are a donator on AA, you can use your Key to speed up downloads and bypass the wait times.
**`USE_CF_BYPASS`:** If disabling the cloudflare bypass, you will be using alternative download hosts.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

**Proxy Configuration:**

```bash
# Basic proxy
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# Proxy with authentication
HTTP_PROXY=http://username:password@proxy.example.com:8080
HTTPS_PROXY=http://username:password@proxy.example.com:8080
```

**`CUSTOM_DNS` Configuration:**

1.  **Custom DNS Servers:** A comma-separated list of DNS server IP addresses.
    Example: `127.0.0.53,127.0.1.53`
2.  **Preset DNS Providers**:
    -   `google`
    -   `quad9`
    -   `cloudflare`
    -   `opendns`
    For more secure DNS providers try these settings:
```bash
CUSTOM_DNS=cloudflare
USE_DOH=true
```

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

**`CUSTOM_SCRIPT`:** The script is called with the full path of the downloaded file as its argument, allowing custom processing like format conversion or validation. The file will be moved to `/cwa-book-ingest` after the script execution.

You can specify these configuration in this format :

```
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

**Note:** If your library volume is on a cifs share, you will get a "database locked" error until you add **nobrl** to your mount line in your fstab file.

Mount should align with your Calibre-Web-Automated ingest folder.

## Variants

### 🧅 Tor Variant

This variant routes all traffic through the Tor network for enhanced privacy.

1.  **Get `docker-compose.tor.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important for Tor:**
*   **Capabilities:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   **Timezone:** The container will attempt to determine the timezone based on the Tor exit node's IP address and set it automatically.
*   **Network Settings:** Custom DNS, DoH, and HTTP(S) proxy settings are ignored.

### External Cloudflare Resolver Variant

Leverage external services like FlareSolverr for advanced Cloudflare bypass.

#### How it Works

-   Requests needing Cloudflare bypass are sent to your external resolver service.
-   The application communicates with the resolver using its API.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

#### To use the External Cloudflare resolver variant:

1.  **Get `docker-compose.extbp.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Important

This feature follows the same configuration of the built-in Cloudflare bypasser, so you should turn on the `USE_CF_BYPASS` configuration to enable it.

#### Compatibility

This feature is designed to work with any resolver that implements the `FlareSolverr` API schema, including `ByParr` and similar projects.

#### Benefits:

- Centralizes Cloudflare bypass logic for easier maintenance.
- Can leverage more powerful or distributed resolver infrastructure.
- Reduces load on the main application container.

## Architecture

The application consists of a single service:

1.  **calibre-web-automated-bookdownloader:** Web interface and download functionality.

## Health Monitoring

Built-in health checks monitor:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

## Logging

*   **Container:** `/var/logs/cwa-book-downloader.log`
*   **Docker Logs:** Access via `docker logs`.

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

MIT License - See the [LICENSE](LICENSE) file.

## Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only.  Users are responsible for:

*   Having the right to download requested materials
*   Respecting copyright laws.
*   Using the tool in compliance with local regulations.

### Duplicate Downloads Warning

The current version:

*   Does not check for existing files
*   Does not verify if books already exist in your Calibre database

Exercise caution when requesting multiple books to avoid duplicates.

## Support

File an issue on the GitHub repository for questions or issues.
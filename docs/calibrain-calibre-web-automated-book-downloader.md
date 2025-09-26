# Automate Your eBook Library with Calibre-Web Automated Book Downloader

[Calibre-Web Automated Book Downloader](https://github.com/calibrain/calibre-web-automated-book-downloader) is your all-in-one solution for seamlessly searching, downloading, and integrating ebooks into your Calibre library.

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request books with a user-friendly web interface.
*   🔄 **Automated Downloads:**  Automatically download books directly to your designated ingest folder.
*   🔌 **Calibre-Web Integration:** Seamlessly integrates with Calibre-Web Automated for a streamlined workflow.
*   📖 **Multiple Format Support:** Supports a wide range of ebook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Bypasses Cloudflare for reliable book downloads.
*   🐳 **Docker Deployment:** Easy setup and deployment with Docker.
*   🧅 **Tor Variant:** Includes a Tor variant for enhanced privacy and bypassing network restrictions.
*   ⚙️ **External Cloudflare Resolver Integration:** Supports integration with external Cloudflare resolver services like FlareSolverr and ByParr for enhanced reliability.

## Getting Started

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

3.  **Access the web interface:** Open your browser and go to `http://localhost:8084`

## Configuration

### Environment Variables

Customize the application behavior using environment variables. Here are the main groups of settings:

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

*   **Authentication:**  Set `CWA_DB_PATH` to point to Calibre-Web's `app.db` to enable authentication.
*   **Logging:** Enabled by default. Logs are stored in `/var/log/cwa-book-downloader`.  Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

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

*   **Multiple Languages:** Specify multiple languages separated by commas in `BOOK_LANGUAGE` (e.g., `en,fr,ru`).

#### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   **Anna's Archive:** Use your Donator key in `AA_DONATOR_KEY` to speed up downloads.  If `USE_CF_BYPASS` is disabled, the app will use alternative download hosts like libgen.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:**  Use `HTTP_PROXY` and `HTTPS_PROXY` to configure proxies, including authentication.
*   **Custom DNS:** Specify custom DNS servers (comma-separated IPs) or predefined providers (`google`, `quad9`, `cloudflare`, `opendns`).  Use `USE_DOH=true` with custom DNS providers for DNS over HTTPS.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Custom Scripts:** Set `CUSTOM_SCRIPT` to run a script after each successful download for custom processing (e.g., format conversion).  The script receives the full path of the downloaded file as an argument and MUST preserve the original filename.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   **Important:**  Ensure your `/your/local/path` volume aligns with your Calibre-Web-Automated ingest folder. If using a cifs share, add `nobrl` to your mount line in your fstab file to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

This variant routes all traffic through the Tor network for enhanced privacy.

1.  **Get the Tor-specific `docker-compose.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Tor Considerations:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Timezone is auto-determined based on Tor exit node IP.  Custom DNS, DoH, and proxy settings are ignored.

### External Cloudflare Resolver Variant

Use an external resolver service (like FlareSolverr) to bypass Cloudflare.

1.  **Get the `docker-compose.extbp.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   **Configuration:**
    *   `EXT_BYPASSER_URL`: The full URL of your resolver.
    *   `EXT_BYPASSER_PATH`: API path for the resolver (usually `/v1`).
    *   `EXT_BYPASSER_TIMEOUT`:  Timeout for page loading (in milliseconds).
*   Requires `USE_CF_BYPASS=true`.

## Architecture

The application consists of a single service:

1.  **calibre-web-automated-bookdownloader**: The main application providing the web interface and download functionality.

## Health Monitoring

Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection.  Checks run every 30 seconds.  See the `HEALTHCHECK` example in the original README for details.

## Logging

*   **Container Logs:** `/var/logs/cwa-book-downloader.log`
*   **Docker Logs:** Access via `docker logs`.

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

This project is licensed under the MIT License.

## Important Disclaimers

*   **Copyright:** Users are responsible for ensuring they have the right to download materials and must comply with copyright laws.
*   **Duplicate Downloads:**  This tool does NOT check for existing files in the download directory or in your Calibre database. Avoid requesting multiple books to avoid duplicates.
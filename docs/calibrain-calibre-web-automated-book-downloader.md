# Effortlessly Download Books for Your Calibre Library with Calibre-Web-Automated Book Downloader

Tired of manually downloading and importing books into your Calibre library? **Calibre-Web-Automated Book Downloader** provides a user-friendly web interface for searching and downloading books, seamlessly integrating with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated-book-downloader) to automate your ebook workflow.

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request book downloads.
*   🔄 **Automated Downloads:** Automatically downloads books to your specified ingest folder.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated.
*   📖 **Multi-Format Support:** Downloads books in various formats including epub, mobi, and more.
*   🛡️ **Cloudflare Bypass:** Built-in Cloudflare bypass functionality for reliable downloads.
*   🐳 **Dockerized Deployment:** Easy setup with Docker Compose.
*   🧅 **Tor Variant:** Option to route all traffic through the Tor network for enhanced privacy.
*   🔌 **External Cloudflare Resolver Variant:** Integration with external Cloudflare resolver services like FlareSolverr.

## Getting Started:

### Prerequisites:

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation:

1.  **Get the `docker-compose.yml` file:**
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**
    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:** Open your browser and navigate to `http://localhost:8084`.

## Configuration:

Customize the application's behavior through environment variables.

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

### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

### Custom Configuration
| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

## Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```
**Note** - If your library volume is on a cifs share, you will get a "database locked" error until you add **nobrl** to your mount line in your fstab file. e.g. //192.168.1.1/Books /media/books cifs credentials=.smbcredentials,uid=1000,gid=1000,iocharset=utf8,**nobrl** - See https://github.com/crocodilestick/Calibre-Web-Automated/issues/64#issuecomment-2712769777

## Variants

### 🧅 Tor Variant

For enhanced privacy, use the Tor variant:

1.  Get the Tor-specific `docker-compose.yml` file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start the service:
    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

### External Cloudflare resolver variant

To leverage an external resolver (like FlareSolverr):

1.  Get the extbp-specific `docker-compose.yml` file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service:
    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Configuration
| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

## Architecture:

*   A single service: `calibre-web-automated-bookdownloader`

## Health Monitoring:

The application has built-in health checks.

## Logging:

Logs are available in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Access via `docker logs`

## Contributing:

Contributions are welcome! Submit a Pull Request.

## License:

This project is licensed under the MIT License.

## Important Disclaimers:

### Copyright Notice

Users are responsible for complying with copyright laws.

### Duplicate Downloads Warning

The current version does not check for existing books.

## Support:

Report issues on the GitHub repository.
# 📚 Automate Your eBook Library: Calibre-Web-Automated Book Downloader

**Effortlessly search, download, and organize your eBooks with a user-friendly web interface designed to seamlessly integrate with your Calibre library.** [Check out the original repository here](https://github.com/calibrain/calibre-web-automated-book-downloader).

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request books for download.
*   🔄 **Automated Downloads:**  Downloads directly to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Multiple Format Support:** Download books in various formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capabilities for reliable downloads, and also the options for external resolver, and Tor variant.
*   🐳 **Docker Deployment:** Simple and fast setup with Docker.

## 🖼️ Screenshots

![Main search interface Screenshot](README_images/search.png 'Main search interface')

![Details modal Screenshot placeholder](README_images/details.png 'Details modal')

![Download queue Screenshot placeholder](README_images/downloading.png 'Download queue')

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

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

**Authentication:** To enable authentication, set `CWA_DB_PATH` to your Calibre-Web's `app.db`.

**Logging:** If logging is enabled, logs are stored in `/var/log/cwa-book-downloader`. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

**Timezone:** When using TOR, the TZ is calculated automatically based on IP.

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

**BOOK_LANGUAGE:** You can specify multiple languages, comma separated (e.g., `en,fr,ru`).

#### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

**AA_DONATOR_KEY:**  Use your key for faster downloads if you are an Anna's Archive donor.

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

**CUSTOM_DNS Options:**

1.  Comma-separated list of DNS server IPs (e.g., `127.0.0.53,127.0.1.53`)
2.  Preset DNS providers: `google`, `quad9`, `cloudflare`, `opendns`

Using alternative DNS providers (e.g., Cloudflare) may help bypass ISP blocks.
You can combine `CUSTOM_DNS` with `USE_DOH=true` for DNS over HTTPS (only supported for google, quad9, cloudflare and opendns)

Try something like this :
```bash
CUSTOM_DNS=cloudflare
USE_DOH=true
```

#### Custom configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

If `CUSTOM_SCRIPT` is set, it will be executed after each successful download but before the file is moved to the ingest directory. This allows for custom processing like format conversion or validation.

The script is called with the full path of the downloaded file as its argument. Important notes:
- The script must preserve the original filename for proper processing
- The file can be modified or even deleted if needed
- The file will be moved to `/cwa-book-ingest` after the script execution (if not deleted)

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

**Note:** If your library volume is on a cifs share, add `nobrl` to your mount line in fstab to avoid "database locked" errors.  See https://github.com/crocodilestick/Calibre-Web-Automated/issues/64#issuecomment-2712769777

## Variants:

### 🧅 Tor Variant

*   Runs all traffic through the Tor network for enhanced privacy.
*   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is auto-detected when using Tor.
*   Ignores custom DNS/proxy settings.

1.  Get `docker-compose.tor.yml`:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start with:
    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

### External Cloudflare resolver variant

*   Uses an external service (e.g., FlareSolverr) to bypass Cloudflare.
*   Improves reliability and performance.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  Get `docker-compose.extbp.yml`:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start with:
    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

## 🏗️ Architecture

*   Single service: `calibre-web-automated-bookdownloader` - Provides the web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass.
*   Checks run every 30 seconds with a 30-second timeout and 3 retries.
```
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pyrequests http://localhost:8084/request/api/status || exit 1
```

## 📝 Logging

*   Logs available in container: `/var/logs/cwa-book-downloader.log`
*   Docker logs can be accessed via `docker logs`.

## 🤝 Contributing

*   Contributions are welcome! Submit a Pull Request.

## 📄 License

*   MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

*   This tool is for legitimate use only. Users are responsible for respecting copyright laws.

### Duplicate Downloads Warning

*   The current version does not check for existing files or Calibre database entries.
*   Exercise caution to avoid duplicate downloads.

## 💬 Support

*   File issues on the GitHub repository for any questions or problems.
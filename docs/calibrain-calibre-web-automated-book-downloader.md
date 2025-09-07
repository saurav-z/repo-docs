# 📚 Automate Your eBook Library with Calibre-Web Automated Book Downloader

Tired of manually downloading eBooks? **Calibre-Web Automated Book Downloader is your all-in-one solution for effortlessly searching, requesting, and downloading eBooks directly to your Calibre library!**  [Check out the original repo](https://github.com/calibrain/calibre-web-automated-book-downloader).

## ✨ Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request books with a user-friendly web interface.
*   🔄 **Automated Downloads:** Downloads directly to your specified ingest folder, ready for your Calibre library.
*   🔌 **Seamless Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Format Support:** Supports a wide range of eBook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes a built-in Cloudflare bypass to ensure reliable downloads, with options for external resolvers.
*   🐳 **Docker-Based:** Quick and easy setup with Docker and Docker Compose.
*   🧅 **Tor Variant**: Option to run all traffic through the Tor network for enhanced privacy.
*   ✅ **Health Monitoring:** Built-in health checks to monitor service availability.

## 🚀 Quick Start:

### Prerequisites:

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation:

1.  Get the `docker-compose.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  Start the service:

    ```bash
    docker compose up -d
    ```

3.  Access the web interface at `http://localhost:8084`

## ⚙️ Configuration:

Customize your experience with various environment variables:

### Application Settings:

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

*   `CWA_DB_PATH`: Set to your Calibre-Web's `app.db` for authentication.
*   Logging: Log files are located in `/var/log/cwa-book-downloader` when enabled. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Download Settings:

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

*   `BOOK_LANGUAGE`: Set multiple comma-separated languages (e.g., `en,fr,ru`).

### AA (Anna's Archive) Settings:

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   `AA_DONATOR_KEY`:  Use your key for faster downloads if you're an AA donator.
*   `USE_CF_BYPASS`: Disable Cloudflare bypass to use alternative download sources.

### Network Settings:

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Proxy Configuration:  Use standard `HTTP_PROXY` and `HTTPS_PROXY` variables, including authentication (e.g., `http://username:password@proxy.example.com:8080`).
*   `CUSTOM_DNS`:  Set DNS servers (comma-separated IP addresses) or use preset DNS providers like `google`, `quad9`, `cloudflare`, or `opendns`.
*   `USE_DOH`:  Use DNS over HTTPS (with a configured preset DNS provider).

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   `CUSTOM_SCRIPT`:  Run a custom script after each download for post-processing.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** Ensure `nobrl` is added to your fstab entry for CIFS shares to avoid "database locked" errors.

## Variants:

### 🧅 Tor Variant:

Run all traffic through the Tor network for enhanced privacy.

1.  Get the Tor-specific docker-compose file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start the service using this file:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Important:**  Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Custom DNS, DoH, and proxy settings are ignored in Tor mode.

### External Cloudflare Resolver Variant:

Leverage an external service (like FlareSolverr) for Cloudflare bypass.

#### Configuration:

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  Get the extbp-specific docker-compose file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service using this file:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

## 🏗️ Architecture:

*   Single Service:  `calibre-web-automated-bookdownloader` - Manages the web interface and download functionality.

## 🏥 Health Monitoring:

*   Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass.

## 📝 Logging:

*   Logs are available in: `/var/logs/cwa-book-downloader.log` (inside the container) or via `docker logs`.

## 🤝 Contributing:

Contributions are welcome!  Submit a Pull Request.

## 📄 License:

MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers:

### Copyright Notice:

*   This tool is designed for **legal and responsible use** only.
*   Users are responsible for their use of the tool and ensuring they respect copyright laws and intellectual property rights.

### Duplicate Downloads Warning:

*   The current version **does not check for duplicate files or verify existing books in your Calibre database**. Exercise caution and manage your downloads.

## 💬 Support:

For issues or questions, please file an issue on the GitHub repository.
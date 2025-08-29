# 📚 Automate Your eBook Library with Calibre-Web Automated Book Downloader

**Tired of manually downloading eBooks?** Automate your Calibre library with the Calibre-Web Automated Book Downloader, a user-friendly web interface for searching and seamlessly downloading books directly to your Calibre library. [See the original repo](https://github.com/calibrain/calibre-web-automated-book-downloader).

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request books.
*   🔄 **Automated Downloads:** Downloads directly to your configured ingest folder.
*   🔌 **Calibre-Web Integration:** Seamlessly integrates with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) for a complete eBook management workflow.
*   📖 **Multiple Format Support:** Supports common eBook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capabilities for reliable downloads.
*   🐳 **Docker Deployment:** Easy setup and management with Docker.
*   🧅 **Tor Variant:** Option to route all traffic through the Tor network for enhanced privacy and bypassing network restrictions.
*   🚀 **External Cloudflare resolver variant**: Integrates with external Cloudflare resolvers for improved reliability and performance.

## 🚀 Quick Start (Docker)

### Prerequisites

*   Docker
*   Docker Compose
*   Running [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) instance (recommended)

### Installation

1.  **Get the Docker Compose file:**
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```
2.  **Start the service:**
    ```bash
    docker compose up -d
    ```
3.  **Access the Web Interface:** Open `http://localhost:8084` in your browser.

## ⚙️ Configuration

### Environment Variables

Configure the application using the following environment variables.  Authentication can be enabled by setting `CWA_DB_PATH` to point to Calibre-Web's `app.db`.

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

#### Anna's Archive Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

#### Custom configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

### Volume Configuration
Mount should align with your Calibre-Web-Automated ingest folder.

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```
**Note** - If your library volume is on a cifs share, you will get a "database locked" error until you add **nobrl** to your mount line in your fstab file. e.g. //192.168.1.1/Books /media/books cifs credentials=.smbcredentials,uid=1000,gid=1000,iocharset=utf8,**nobrl** - See https://github.com/crocodilestick/Calibre-Web-Automated/issues/64#issuecomment-2712769777

### Tor Variant

For enhanced privacy and network restriction bypass, use the Tor variant.

1.  Get the Tor-specific docker-compose file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start the service using this file:
    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

### External Cloudflare resolver variant

This variant allows the application to use an external service to bypass Cloudflare protection.

1.  Get the extbp-specific docker-compose file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service using this file:
    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

#### Important
This feature follows the same configuration of the built-in Cloudflare bypasser, so you should turn on the `USE_CF_BYPASS` configuration to enable it.

## 🏗️ Architecture

*   **calibre-web-automated-bookdownloader:** Main application providing web interface and download functionality

## 🏥 Health Monitoring

*   Built-in health checks every 30 seconds to monitor service availability.

## 📝 Logging

*   Logs are available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

Contributions are welcome!

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Important Disclaimers

*   **Copyright Notice:** Use responsibly and ensure you have the right to download materials.  The tool is for legitimate use only.
*   **Duplicate Downloads:** The current version doesn't prevent duplicate downloads.

## 💬 Support

For issues or questions, please file an issue on the GitHub repository.
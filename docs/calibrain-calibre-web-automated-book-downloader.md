# 📚 Effortlessly Download Books for Your Calibre Library with Calibre-Web-Automated Book Downloader

Streamline your ebook workflow and easily download books for your Calibre library with the user-friendly [Calibre-Web-Automated Book Downloader](https://github.com/calibrain/calibre-web-automated-book-downloader).

**Key Features:**

*   🌐 **Intuitive Web Interface:** Search and request book downloads with ease.
*   🔄 **Automated Downloads:** Directly download books to your specified ingest folder.
*   🔌 **Seamless Integration:** Works perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Multiple Format Support:** Compatible with epub, mobi, azw3, fb2, djvu, cbz, and cbr formats.
*   🛡️ **Cloudflare Bypass:** Reliable downloads with built-in Cloudflare bypass capabilities.
*   🐳 **Dockerized Deployment:** Quick and easy setup with Docker.

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

1.  Get the docker-compose.yml:

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

*   Enable authentication by setting `CWA_DB_PATH` to your Calibre-Web's `app.db` path.
*   If logging is enabled, logs default to `/var/log/cwa-book-downloader`. Available log levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
*   Note that if using TOR, the TZ will be calculated automatically based on IP.

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

*   Set `BOOK_LANGUAGE` with comma-separated values like `en,fr,ru` for multiple language preferences.

#### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   AA Donators can use their key in `AA_DONATOR_KEY` for faster downloads.
*   If disabling the cloudflare bypass, you will be using alternative download hosts, such as libgen or z-lib, but they usually have a delay before getting the more recent books and their collection is not as big as aa's. But this setting should work for the majority of books.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Proxy settings: Use `HTTP_PROXY` and `HTTPS_PROXY` for basic or authenticated proxies.
*   `CUSTOM_DNS`: Configure custom DNS servers (comma-separated IPs) or use preset DNS providers: `google`, `quad9`, `cloudflare`, or `opendns`.  Use `USE_DOH=true` with a preset DNS provider to force DNS over HTTPS.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   Use `CUSTOM_SCRIPT` to run a custom script after each download for processing. The script receives the downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   **Note:** If your library volume is on a cifs share, you will get a "database locked" error until you add **nobrl** to your mount line in your fstab file.
*   Mount should align with your Calibre-Web-Automated ingest folder.

## Variants

### 🧅 Tor Variant

Use the Tor variant for enhanced privacy:

1.  Get the Tor-specific docker-compose file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  Start the service using this file:

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Considerations for Tor:**

*   This variant requires the `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is determined by the Tor exit node's IP.
*   Custom DNS, DoH, and proxy settings are ignored.

### External Cloudflare Resolver Variant

Use an external service to bypass Cloudflare protection.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

#### To use the External Cloudflare resolver variant:

1.  Get the extbp-specific docker-compose file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  Start the service using this file:

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Compatibility:

This feature is designed to work with any resolver that implements the `FlareSolverr` API schema, including `ByParr` and similar projects.

#### Benefits:

- Centralizes Cloudflare bypass logic for easier maintenance.
- Can leverage more powerful or distributed resolver infrastructure.
- Reduces load on the main application container.

## 🏗️ Architecture

*   The application consists of a single service:
    *   **calibre-web-automated-bookdownloader**: Main application providing web interface and download functionality

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection. Checks run every 30 seconds with a 30-second timeout and 3 retries.

## 📝 Logging

*   Logs are available in the container: `/var/logs/cwa-book-downloader.log` and via Docker logs.

## 🤝 Contributing

Contributions are welcome! Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool accesses sources which may contain copyrighted material. Users are responsible for ensuring they have the right to download materials, respecting copyright laws, and complying with local regulations.

### Duplicate Downloads Warning

This tool does not check for existing files in the download directory or verify if books already exist in your Calibre database. Exercise caution to avoid duplicates.

## 💬 Support

For issues or questions, file an issue on the GitHub repository.
# 📚 Automate Your eBook Downloads with Calibre-Web Book Downloader

Simplify your eBook acquisition process with the **Calibre-Web Automated Book Downloader**, a user-friendly web interface designed to seamlessly integrate with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated). 

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Easily search and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder.
*   🔌 **Calibre-Web Integration:** Designed to work perfectly with Calibre-Web-Automated.
*   📖 **Format Support:** Download books in various formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capabilities for reliable downloads.
*   🐳 **Docker Deployment:** Easy setup with Docker.
*   🧅 **Tor Integration:** Provides a Tor variant for enhanced privacy and network bypass.
*   🌉 **External Cloudflare Resolver:** Integrates with external services like FlareSolverr for advanced Cloudflare bypass.

## 🖼️ Screenshots
*(Placeholder for screenshots)*

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation

1.  **Get the Docker Compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:**  `http://localhost:8084`

## ⚙️ Configuration

### Environment Variables

Configure your book downloads with these environment variables:

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
*   **Logging:** Logs are stored in `/var/log/cwa-book-downloader` if logging is enabled. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

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

*   **AA Donator Key:** Enter your key in `AA_DONATOR_KEY` for faster downloads.
*   **Cloudflare Bypass:** Disable bypass by setting `USE_CF_BYPASS` to `false`.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:** Use `HTTP_PROXY` and `HTTPS_PROXY` for proxy settings (with or without authentication).
*   **Custom DNS:** Set `CUSTOM_DNS` to a comma-separated list of DNS server IPs or use preset providers like `google`, `quad9`, `cloudflare`, or `opendns`.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   **Custom Script:** Set `CUSTOM_SCRIPT` to execute a script after each download.  The script receives the downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   **Note:**  For CIFS shares, include **nobrl** in your fstab mount line to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

For enhanced privacy, use the Tor variant to route all traffic through the Tor network.

1.  **Get the Tor docker-compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Important:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Timezone is based on Tor exit node IP.  Network settings (DNS, DoH, proxies) are ignored.

### 🌉 External Cloudflare Resolver Variant

Integrate with external Cloudflare resolvers like FlareSolverr for improved reliability.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  **Get the extbp docker-compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   **Compatibility:** Works with resolvers using the FlareSolverr API schema.
*   **Enable:** Set `USE_CF_BYPASS` to true.

## 🏗️ Architecture

*   The application consists of a single service: `calibre-web-automated-bookdownloader`

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass.

## 📝 Logging

*   Logs are available in the container at `/var/logs/cwa-book-downloader.log` and via Docker logs.

## 🤝 Contributing

*   Contributions are welcome!  Submit pull requests.

## 📄 License

*   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## ⚠️ Important Disclaimers

### Copyright Notice

*   This tool is designed for **legal and ethical use only**.  Users are responsible for complying with copyright laws and intellectual property rights.

### Duplicate Downloads Warning

*   The current version **does not prevent duplicate downloads.**  Be careful when requesting books.

## 💬 Support

*   For issues or questions, please [file an issue](https://github.com/calibrain/calibre-web-automated-book-downloader/issues) on the GitHub repository.
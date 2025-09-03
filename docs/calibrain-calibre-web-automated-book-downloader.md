# 📚 Automate Your eBook Library: Calibre-Web-Automated Book Downloader

**Streamline your eBook workflow with the Calibre-Web-Automated Book Downloader, a user-friendly web interface for effortlessly searching, downloading, and organizing books for your Calibre library. ([Original Repository](https://github.com/calibrain/calibre-web-automated-book-downloader))**

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Search and download books with ease.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Multi-Format Support:** Supports popular eBook formats like epub, mobi, azw3, and more.
*   🛡️ **Cloudflare Bypass:** Bypasses Cloudflare protection for reliable downloads.
*   🐳 **Dockerized Deployment:** Easy setup and management with Docker.
*   🧅 **Tor Variant:** Optional Tor variant for enhanced privacy and bypassing network restrictions.
*   💻 **External Cloudflare Resolver Support:** Integrates with services like FlareSolverr for a more robust bypass solution.

## 🚀 Getting Started

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

3.  **Access the web interface:**  `http://localhost:8084`

## ⚙️ Configuration

### Environment Variables

Configure the application using environment variables.

#### Application Settings

| Variable          | Description                | Default Value      |
| ----------------- | -------------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port         | `8084`             |
| `FLASK_HOST`      | Web interface binding      | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle          | `false`            |
| `INGEST_DIR`      | Book download directory    | `/cwa-book-ingest` |
| `TZ`              | Container timezone         | `UTC`              |
| `UID`             | Runtime user ID            | `1000`             |
| `GID`             | Runtime group ID           | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database     | None               |
| `ENABLE_LOGGING`  | Enable log file            | `true`             |
| `LOG_LEVEL`       | Log level to use           | `info`             |

*   `CWA_DB_PATH`:  Required if you wish to enable authentication; points to Calibre-Web's `app.db`.
*   Logging: Logs are saved to `/var/log/cwa-book-downloader` if enabled.  Available levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

#### Download Settings

| Variable               | Description                                                 | Default Value                     |
| ---------------------- | ----------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                      | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                       | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                             | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                      | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                              | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                  | `false`                           |
| `PRIORITIZE_WELIB`     | Download from WELIB first instead of AA                    | `false`                           |

*   `BOOK_LANGUAGE`:  Supports multiple languages separated by commas (e.g., `en,fr,ru`).

#### AA (Anna's Archive) Settings

| Variable               | Description                                                 | Default Value                     |
| ---------------------- | ----------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   `AA_DONATOR_KEY`: Enter your Anna's Archive Donator Key for faster downloads.
*   `USE_CF_BYPASS`: Disable Cloudflare Bypass to use alternative download hosts (libgen, z-lib).

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Proxy Configuration:  Specify proxies using `HTTP_PROXY` and `HTTPS_PROXY`.  Authentication is supported.
*   `CUSTOM_DNS`:  Can be a comma-separated list of DNS servers or predefined providers (`google`, `quad9`, `cloudflare`, `opendns`).

#### Custom Configuration

| Variable          | Description                                            | Default Value           |
| ----------------- | ------------------------------------------------------ | ----------------------- |
| `CUSTOM_SCRIPT`   | Path to an executable script after each download     | ``                      |

*   `CUSTOM_SCRIPT`:  Executes a script after each download for custom processing (e.g., format conversion). The script receives the downloaded file's full path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   Mount the ingest folder to your desired location.
*   If using a CIFS share, add **nobrl** to your fstab mount line to avoid "database locked" errors.

## 🌐 Variants

### 🧅 Tor Variant

Run the application through the Tor network for enhanced privacy.

1.  **Get the `docker-compose.tor.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

    *   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Timezone is determined by the Tor exit node.  Custom DNS, DoH, and proxy settings are ignored.

### 💻 External Cloudflare Resolver Variant

Integrate with an external Cloudflare resolver service (like FlareSolverr) for improved bypass performance and reliability.

1.  **Get the `docker-compose.extbp.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

*   Ensure `USE_CF_BYPASS` is enabled to utilize the external resolver.

## 🏗️ Architecture

*   A single service:  `calibre-web-automated-bookdownloader`

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection.

## 📝 Logging

*   Logs are available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs:  Access using `docker logs`

## 🤝 Contributing

Contributions are welcome!  Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

### Copyright Notice

*   This tool is designed for legitimate use only.  Users are responsible for complying with copyright laws and ensuring they have the right to download requested materials.

### Duplicate Downloads Warning

*   The current version does not check for existing files in the download directory or verify if books already exist in your Calibre database.

## 💬 Support

*   Report issues or ask questions on the GitHub repository.
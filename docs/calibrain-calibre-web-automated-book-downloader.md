# Download and Organize Your eBooks with Calibre-Web Automated Book Downloader

**Tired of manually downloading and organizing your eBooks?** Calibre-Web Automated Book Downloader simplifies the process with a user-friendly interface, automated downloads, and seamless integration with Calibre-Web-Automated. Access the original repository [here](https://github.com/calibrain/calibre-web-automated-book-downloader).

## ✨ Key Features

*   🌐 **Intuitive Web Interface:** Easily search for and request book downloads.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Wide Format Support:** Downloads multiple ebook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Built-in capability to bypass Cloudflare for reliable downloads.
*   🐳 **Dockerized Deployment:** Easy setup and management with Docker and Docker Compose.

## 🖼️ Screenshots

*(Placeholder - Add screenshots here)*

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

3.  **Access the web interface:** Browse to `http://localhost:8084`

## ⚙️ Configuration

Configure the behavior of the application using the environment variables described below.

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

**Authentication:** To enable authentication, set `CWA_DB_PATH` to point to Calibre-Web's `app.db`.

**Logging:** Logs are stored in `/var/log/cwa-book-downloader`.  Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

**Timezone:**  When using Tor, the timezone will be automatically determined based on your IP address.

#### Download Settings

| Variable               | Description                                                 | Default Value                     |
| ---------------------- | ----------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                    | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                     | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                           | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                    | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                              | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                  | `false`                           |
| `PRIORITIZE_WELIB`     | Download from WELIB first instead of AA                   | `false`                           |

**Multiple Languages:**  Set `BOOK_LANGUAGE` to a comma-separated list (e.g., `en,fr,ru`).

#### Anna's Archive (AA) Settings

| Variable               | Description                                                 | Default Value                    |
| ---------------------- | ----------------------------------------------------------- | -------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)       | `https://annas-archive.org`      |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                           |

**AA Donator Key:** Provide your key in `AA_DONATOR_KEY` for faster downloads.

**Cloudflare Bypass:** Disable `USE_CF_BYPASS` to use alternative download hosts.

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
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
HTTP_PROXY=http://username:password@proxy.example.com:8080
HTTPS_PROXY=http://username:password@proxy.example.com:8080
```

**Custom DNS:**

*   Comma-separated IPs: `127.0.0.53,127.0.1.53`
*   Preset providers: `google`, `quad9`, `cloudflare`, `opendns`

Use DNS over HTTPS (DoH) by setting `USE_DOH=true` when using a preset provider.

**Example:**

```bash
CUSTOM_DNS=cloudflare
USE_DOH=true
```

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

**Custom Script Execution:**  Run a script after each download for custom processing.  The script receives the file path as an argument.

**Example:**

```yaml
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

**Important:**  If using a CIFS share, add `nobrl` to your mount options in `/etc/fstab` to avoid "database locked" errors.

## 🔄 Variants

### 🧅 Tor Variant

Run all traffic through the Tor network for enhanced privacy.

1.  Get the Tor-specific compose file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  Start the service:
    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Tor Considerations:**  Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Timezone and network settings are automatically handled in Tor mode.

### External Cloudflare Resolver Variant

Integrate with an external Cloudflare resolver (like FlareSolverr) for improved reliability.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  Get the external resolver compose file:
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```
2.  Start the service:
    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

**Compatibility:** Works with any resolver that implements the FlareSolverr API.

## 🏗️ Architecture

*   **calibre-web-automated-bookdownloader:** The main application providing the web interface and download functionality.

## 🏥 Health Monitoring

Built-in health checks monitor the web interface, download service, and Cloudflare bypass service.

## 📝 Logging

Logs are available in the container at `/var/logs/cwa-book-downloader.log` and via Docker logs.

## 🤝 Contributing

Contributions are welcome! Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is for legitimate use only. Users are responsible for ensuring they have the right to download materials and must respect copyright laws and local regulations.

### Duplicate Downloads

The current version does not check for existing files or Calibre database entries.  Exercise caution to avoid duplicates.

## 💬 Support

File issues on the GitHub repository for support.
# 📚 Automate Your eBook Library with Calibre-Web Book Downloader

**Effortlessly search, download, and integrate eBooks into your Calibre library with this intuitive web interface.** [Learn more at the original repository](https://github.com/calibrain/calibre-web-automated-book-downloader).

## ✨ Key Features

*   🌐 **User-Friendly Web Interface:** Easily search for and request book downloads.
*   🔄 **Automated Downloads:** Automatically downloads books to your specified ingest directory.
*   🔌 **Seamless Integration:** Designed to work perfectly with Calibre-Web-Automated.
*   📖 **Multi-Format Support:** Compatible with common eBook formats like epub, mobi, azw3, and more.
*   🛡️ **Cloudflare Bypass:** Bypasses Cloudflare protection for reliable downloads.
*   🐳 **Docker Deployment:** Quick and easy setup with Docker.
*   🧅 **Tor Variant:** Offers a Tor-enabled version for enhanced privacy.
*   🤖 **External Cloudflare Resolver Support:** Integrate with services like FlareSolverr or ByParr.

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

1.  Get the `docker-compose.yml` file:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  Start the service:

    ```bash
    docker compose up -d
    ```

3.  Access the web interface at `http://localhost:8084`

## ⚙️ Configuration

Configure the application using environment variables.

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

*   To enable authentication, set `CWA_DB_PATH` to point to Calibre-Web's `app.db`.
*   Log folder is `/var/log/cwa-book-downloader` if logging is enabled.

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

*   Set multiple languages for `BOOK_LANGUAGE` separated by commas.

### Anna's Archive Settings (AA)

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

*   Use your AA donator key with `AA_DONATOR_KEY` for faster downloads.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   Configure proxies using the format: `HTTP_PROXY=http://username:password@proxy.example.com:8080`.
*   Use custom DNS servers (comma separated) or preset providers: `google`, `quad9`, `cloudflare`, `opendns`. Set `USE_DOH=true` to use DNS over HTTPS with supported providers.

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

*   Use `CUSTOM_SCRIPT` to run a script after each download. The script receives the downloaded file path as an argument.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   Mount your ingest directory and Calibre-Web's database.
*   If using a CIFS share, add **nobrl** to your fstab mount options.

## Variants

### 🧅 Tor Variant

*   Use a Tor-specific Docker Compose file: `docker-compose.tor.yml`. Requires `NET_ADMIN` and `NET_RAW` Docker capabilities. Timezone is set based on Tor exit node.
*   Network settings are ignored in the Tor variant.

### 🤖 External Cloudflare Resolver Variant

*   Use a specific Docker Compose file: `docker-compose.extbp.yml`.
*   Integrate with services like FlareSolverr or ByParr to handle Cloudflare bypass.

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

*   Enable by setting `USE_CF_BYPASS=true`.

## 🏗️ Architecture

The application comprises a single service:

1.  **calibre-web-automated-bookdownloader:** Provides the web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass service.
*   Checks run every 30 seconds with 3 retries.
*   You can add a `HEALTHCHECK` to your compose.

## 📝 Logging

*   Logs are available in `/var/logs/cwa-book-downloader.log` inside the container.
*   Access Docker logs using the `docker logs` command.

## 🤝 Contributing

Contributions are welcome! Submit pull requests on GitHub.

## 📄 License

This project is licensed under the MIT License (see the [LICENSE](LICENSE) file).

## ⚠️ Important Disclaimers

### Copyright Notice

*   This tool is for legitimate use only. Users are responsible for complying with copyright laws.

### Duplicate Downloads Warning

*   The current version does not check for existing files or Calibre database entries.
*   Exercise caution to avoid downloading duplicates.

## 💬 Support

*   For issues and questions, please file an issue on the GitHub repository.
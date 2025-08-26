# Automate Your Book Downloads with Calibre-Web Automated Book Downloader

**Tired of manually searching for books? Calibre-Web Automated Book Downloader is a user-friendly web interface that automates book downloads and seamlessly integrates with Calibre-Web-Automated, simplifying your ebook library management.**  [Explore the original project on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader).

## Key Features

*   🌐 **Intuitive Web Interface:** Easily search and request book downloads with a clean and user-friendly interface.
*   🔄 **Automated Downloads:** Automatically downloads books to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work flawlessly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   📖 **Multiple Format Support:** Download a wide range of ebook formats, including epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   🛡️ **Cloudflare Bypass:** Includes a Cloudflare bypass for reliable access to book sources.
*   🐳 **Docker-Based Deployment:** Quick and easy setup with Docker, ensuring portability and ease of use.
*   🧅 **Tor Integration (Optional):** Provides a Tor variant for enhanced privacy and bypassing network restrictions.

## Getting Started

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get the Docker Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the Web Interface:**  Open your web browser and navigate to `http://localhost:8084`.

## Configuration

Customize the application's behavior using environment variables:

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

**Note:** To enable authentication, set `CWA_DB_PATH` to your Calibre-Web `app.db`.

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

### Anna's Archive (AA) Settings

| Variable               | Description                                                     | Default Value                    |
| ---------------------- | --------------------------------------------------------------- | -------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)          | `https://annas-archive.org`      |
| `USE_CF_BYPASS`        | Disable Cloudflare bypass and use alternative links instead       | `true`                           |

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

Configure data persistence through Docker volumes:

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

## Tor Variant

For enhanced privacy, utilize the Tor-enabled variant:

1.  **Get the Tor Docker Compose File:**
    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```
2.  **Start the Service:**
    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```
   *Note: This variant requires the NET_ADMIN and NET_RAW Docker capabilities.*

## Architecture

The application consists of a single service:

1.  **calibre-web-automated-bookdownloader:** The primary service, providing the web interface and download functionality.

## Health Monitoring

The application includes built-in health checks to ensure smooth operation, performed every 30 seconds.

## Logging

*   **Container Logs:**  `/var/logs/cwa-book-downloader.log`
*   **Docker Logs:** Access via `docker logs`

## Contributing

We welcome contributions!  Please submit pull requests to help improve this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Copyright Notice

This tool is designed for legitimate use only. Users are responsible for respecting copyright laws and ensuring they have the right to download the requested materials.

### Duplicate Downloads Warning

This application does not currently check for existing files or verify if books already exist in your Calibre database. Please exercise caution to avoid duplicate downloads.

## Support

For any issues or questions, please create an issue on the GitHub repository.
# 📚 Calibre-Web Automated Book Downloader: Effortlessly Download Books for Your Calibre Library

**Simplify your ebook management:** This user-friendly web interface seamlessly integrates with Calibre-Web-Automated, enabling you to easily search for and download books directly to your library. Learn more about this open-source project on [GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader).

## 🌟 Key Features

*   **Intuitive Web Interface:** Easily search and request book downloads.
*   **Automated Downloads:** Books are downloaded directly to your specified ingest folder for easy Calibre integration.
*   **Seamless Integration:** Works perfectly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).
*   **Multi-Format Support:** Supports popular ebook formats: epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   **Cloudflare Bypass:** Includes built-in Cloudflare bypass for reliable access to book sources.
*   **Dockerized Deployment:** Easy setup with Docker for quick deployment.
*   **Tor Integration:** Optional Tor variant for enhanced privacy and network bypassing.

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation Steps

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:**  Open your browser and go to `http://localhost:8084`

## ⚙️ Configuration

### Environment Variables

Configure the application's behavior using environment variables.  Here are some key settings:

#### Application Settings

| Variable          | Description              | Default Value      |
| ----------------- | ------------------------ | ------------------ |
| `FLASK_PORT`      | Web interface port       | `8084`             |
| `FLASK_HOST`      | Web interface binding    | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle        | `false`            |
| `INGEST_DIR`      | Book download directory  | `/cwa-book-ingest` |
| `TZ`              | Container timezone       | `UTC`              |
| `UID`             | Runtime user ID          | `1000`             |
| `GID`             | Runtime group ID         | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database  | None               |
| `ENABLE_LOGGING`  | Enable log file          | `true`             |
| `LOG_LEVEL`       | Log level to use         | `info`             |

*   **Authentication:** Enable authentication by setting `CWA_DB_PATH` to point to your Calibre-Web's `app.db` to match username/password.
*   **Logging:** Logs are stored at `/var/log/cwa-book-downloader` if enabled. Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Tor-Based Timezone:** When running in the Tor variant, the timezone will be determined based on your exit node's IP address.

#### Download Settings

| Variable               | Description                                              | Default Value                     |
| ---------------------- | -------------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                   | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                    | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                          | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                   | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                             | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID                 | `false`                           |
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB first instead of AA | `false`                           |

*   **Multiple Languages:** Specify multiple book languages using comma separation (e.g., `en,fr,ru`).

#### Anna's Archive (AA) Settings

| Variable               | Description                                              | Default Value                       |
| ---------------------- | -------------------------------------------------------- | ----------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)   | `https://annas-archive.org`         |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                              |

*   **AA Donator Key:** Use your AA Donator Key in `AA_DONATOR_KEY` for faster downloads.
*   **CF Bypass:** When disabling CF bypass, you will be using alternative download hosts like libgen and z-lib.

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:** Use the following format to configure proxies:
    ```bash
    HTTP_PROXY=http://proxy.example.com:8080
    HTTPS_PROXY=http://proxy.example.com:8080
    HTTP_PROXY=http://username:password@proxy.example.com:8080
    HTTPS_PROXY=http://username:password@proxy.example.com:8080
    ```
*   **Custom DNS:** Supports custom DNS servers and predefined DNS providers. For example: `CUSTOM_DNS=cloudflare`.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that runs after each download  | ``                      |

*   **Custom Script:** Execute a script after each successful download for processing.
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

*   **CIFS Share:** When using a CIFS share, add **nobrl** to your mount options in `/etc/fstab` to avoid "database locked" errors.

## 🧅 Tor Variant

Utilize the Tor network for enhanced privacy and network bypassing.

1.  **Get the Tor-specific `docker-compose.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Tor Considerations:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Custom DNS, DoH, and proxy settings are ignored with Tor.

## 🏗️ Architecture

*   The application comprises a single service:  `calibre-web-automated-bookdownloader`

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface, download service, and Cloudflare bypass every 30 seconds.

## 📝 Logging

*   Logs are available in the container at `/var/logs/cwa-book-downloader.log` and through `docker logs`.

## 🤝 Contributing

*   Contributions are welcome!  Submit Pull Requests.

## 📄 License

*   MIT License - See the [LICENSE](LICENSE) file.

## ⚠️ Important Disclaimers

### Copyright Notice

*   This tool is designed for legitimate use only. Users are responsible for complying with copyright laws.

### Duplicate Downloads Warning

*   The current version does not check for existing files or duplicates. Exercise caution when requesting multiple books.

## 💬 Support

*   For issues and questions, please file an issue on the GitHub repository.
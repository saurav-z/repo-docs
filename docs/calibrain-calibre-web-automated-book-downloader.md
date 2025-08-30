# Automate Your eBook Library: Calibre-Web-Automated Book Downloader

**Seamlessly find, request, and download books for your Calibre library with this intuitive web interface.  [Get Started](https://github.com/calibrain/calibre-web-automated-book-downloader)**

This project streamlines the process of downloading books and preparing them for seamless integration into your Calibre library, working flawlessly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).

## Key Features

*   **User-Friendly Interface:** Easily search and request book downloads via a clean, modern web interface.
*   **Automated Downloads:** Books are automatically downloaded to your specified ingest folder, ready for Calibre.
*   **Calibre-Web Integration:** Designed for perfect compatibility with Calibre-Web-Automated for a complete ebook management solution.
*   **Multiple Format Support:** Download a wide variety of ebook formats, including epub, mobi, azw3, fb2, djvu, cbz, and cbr.
*   **Cloudflare Bypass:** Includes built-in Cloudflare bypass capability for reliable downloads.
*   **Docker Deployment:**  Quick and easy setup using Docker for portability and ease of use.
*   **Tor Integration:** Included Tor variant for privacy and bypass of network restrictions
*   **External Cloudflare resolver variant:** Supports external services like FlareSolverr for Cloudflare bypass, improving reliability.

## Screenshots

<!-- Add image tags with descriptive alt text and relative links to images -->
<img src="README_images/search.png" alt="Main Search Interface" width="400">
<img src="README_images/details.png" alt="Book Details Modal" width="400">
<img src="README_images/downloading.png" alt="Download Queue" width="400">

## Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get `docker-compose.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the Web Interface:**  Browse to `http://localhost:8084`

## Configuration

### Environment Variables

Configure the application behavior using environment variables.

#### Application Settings

| Variable          | Description                       | Default Value      |
| ----------------- | --------------------------------- | ------------------ |
| `FLASK_PORT`      | Web interface port              | `8084`             |
| `FLASK_HOST`      | Web interface binding           | `0.0.0.0`          |
| `DEBUG`           | Debug mode toggle               | `false`            |
| `INGEST_DIR`      | Book download directory         | `/cwa-book-ingest` |
| `TZ`              | Container timezone              | `UTC`              |
| `UID`             | Runtime user ID                 | `1000`             |
| `GID`             | Runtime group ID                | `100`              |
| `CWA_DB_PATH`     | Calibre-Web's database          | None               |
| `ENABLE_LOGGING`  | Enable log file                 | `true`             |
| `LOG_LEVEL`       | Log level to use                | `info`             |

*   **Authentication:** Set `CWA_DB_PATH` to Calibre-Web's `app.db` to enable authentication.
*   **Logging:** Logs are located in `/var/log/cwa-book-downloader` when `ENABLE_LOGGING` is true. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **Timezone:** If using TOR, the TZ will be calculated automatically based on IP.

#### Download Settings

| Variable               | Description                                         | Default Value                     |
| ---------------------- | --------------------------------------------------- | --------------------------------- |
| `MAX_RETRY`            | Maximum retry attempts                                | `3`                               |
| `DEFAULT_SLEEP`        | Retry delay (seconds)                                 | `5`                               |
| `MAIN_LOOP_SLEEP_TIME` | Processing loop delay (seconds)                       | `5`                               |
| `SUPPORTED_FORMATS`    | Supported book formats                                | `epub,mobi,azw3,fb2,djvu,cbz,cbr` |
| `BOOK_LANGUAGE`        | Preferred language for books                          | `en`                              |
| `AA_DONATOR_KEY`       | Optional Donator key for Anna's Archive fast download API | ``                                |
| `USE_BOOK_TITLE`       | Use book title as filename instead of ID              | `false`                           |
| `PRIORITIZE_WELIB`     | When downloading, download from WELIB instead of AA  | `false`                           |

*   **`BOOK_LANGUAGE`**:  Supports comma-separated language codes (e.g., `en,fr,ru`).

#### AA (Anna's Archive) Settings

| Variable               | Description                                                | Default Value                     |
| ---------------------- | ---------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (can be changed for a proxy)   | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead          | `true`                            |

*   **AA_DONATOR_KEY**: Use your Anna's Archive Donator key for faster downloads.
*   **Cloudflare Bypass:** Disable Cloudflare bypass to use alternative download sources (libgen, z-lib).

#### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

*   **Proxy Configuration:** Use `HTTP_PROXY` and `HTTPS_PROXY` to configure proxies, including authentication.
*   **Custom DNS:** Configure custom DNS servers or use preset providers like Google, Cloudflare, Quad9, or OpenDNS.

#### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that runs after each download | ``                      |

*   **Custom Scripts**:  Execute a custom script after each download for post-processing (e.g., format conversion).  The script receives the full path to the downloaded file as an argument. The downloaded file must not be renamed by the script.

### Volume Configuration

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

*   **`nobrl`**: Required in fstab for CIFS shares to avoid database lock errors. Example: `//192.168.1.1/Books /media/books cifs credentials=.smbcredentials,uid=1000,gid=1000,iocharset=utf8,nobrl`

## Variants

### 🧅 Tor Variant

This variant routes all traffic through the Tor network for enhanced privacy and bypassing network restrictions.

1.  **Get `docker-compose.tor.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

*   **Important:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.  Custom DNS, DoH, and proxy settings are ignored.

### External Cloudflare Resolver Variant

Leverage an external service (e.g., FlareSolverr) for Cloudflare bypass.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

1.  **Get `docker-compose.extbp.yml`:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

*   Requires `USE_CF_BYPASS` to be enabled.

## Architecture

The application architecture consists of a single service:

1.  **calibre-web-automated-bookdownloader**: The core service providing the web interface and download functionality.

## Health Monitoring

Built-in health checks:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

Checks run every 30 seconds, with a 30-second timeout and 3 retries.
Enable using the following within your compose file:
```
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pyrequests http://localhost:8084/request/api/status || exit 1
```

## Logging

Logs are available in:

*   Container: `/var/logs/cwa-book-downloader.log`
*   Docker logs: Access using `docker logs`

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Copyright Notice

This tool accesses sources that may contain copyrighted material.  Users are responsible for:

*   Having the right to download requested materials.
*   Respecting copyright laws and intellectual property rights.
*   Using the tool in compliance with local regulations.

### Duplicate Downloads Warning

*   Does not check for existing files in the download directory.
*   Does not verify if books already exist in your Calibre database.
*   Exercise caution to avoid duplicate downloads.

## Support

For issues or questions, please file an issue on the GitHub repository.
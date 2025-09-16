# 📚 Automate Your eBook Library with Calibre-Web Automated Book Downloader

**Instantly search, download, and seamlessly integrate eBooks into your Calibre library with a user-friendly web interface.** [Learn more and contribute on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader).

## Key Features

*   **Intuitive Web Interface:** Easily search for and request books with a clean, user-friendly design.
*   **Automated Downloads:** Automatically downloads books to your specified ingest folder, ready for Calibre.
*   **Calibre-Web Automated Integration:** Designed for seamless integration with Calibre-Web-Automated.
*   **Format Support:** Supports a wide range of eBook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   **Cloudflare Bypass:** Includes built-in Cloudflare bypass capabilities for reliable downloads.
*   **Docker Deployment:** Simplifies setup with Docker for quick and easy deployment.
*   **Tor Variant:** Supports routing all traffic through the Tor network for enhanced privacy.
*   **External Cloudflare Resolver Variant:** Integrates with external services like FlareSolverr for more robust Cloudflare bypassing.

## Quick Start

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

3.  **Access the web interface:** Open `http://localhost:8084` in your web browser.

## Configuration

### Environment Variables

Customize the behavior of the downloader through environment variables in your `docker-compose.yml`.

#### Application Settings

*   `FLASK_PORT`: Web interface port (default: `8084`)
*   `FLASK_HOST`: Web interface binding (default: `0.0.0.0`)
*   `DEBUG`: Debug mode toggle (default: `false`)
*   `INGEST_DIR`: Book download directory (default: `/cwa-book-ingest`)
*   `TZ`: Container timezone (default: `UTC`)
*   `UID`: Runtime user ID (default: `1000`)
*   `GID`: Runtime group ID (default: `100`)
*   `CWA_DB_PATH`: Calibre-Web's database path. *Required for authentication*.
*   `ENABLE_LOGGING`: Enable log file (default: `true`)
*   `LOG_LEVEL`: Log level to use (default: `info`). Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

If using logging, the log folder defaults to `/var/log/cwa-book-downloader`.

#### Download Settings

*   `MAX_RETRY`: Maximum retry attempts (default: `3`)
*   `DEFAULT_SLEEP`: Retry delay (seconds) (default: `5`)
*   `MAIN_LOOP_SLEEP_TIME`: Processing loop delay (seconds) (default: `5`)
*   `SUPPORTED_FORMATS`: Supported book formats (default: `epub,mobi,azw3,fb2,djvu,cbz,cbr`)
*   `BOOK_LANGUAGE`: Preferred language for books (default: `en`).  Multiple languages separated by commas are supported, e.g., `en,fr,ru`.
*   `AA_DONATOR_KEY`: Anna's Archive (AA) Donator key for faster downloads.
*   `USE_BOOK_TITLE`: Use book title as filename instead of ID (default: `false`)
*   `PRIORITIZE_WELIB`: Download from WELIB first instead of AA (default: `false`)

#### Anna's Archive (AA) Settings

*   `AA_BASE_URL`: Annas-Archive base URL (default: `https://annas-archive.org`)
*   `USE_CF_BYPASS`: Disable CF bypass and use alternative links (default: `true`)

Use your AA Donator Key for faster downloads. If disabling the Cloudflare bypass, you will be using alternative download hosts.

#### Network Settings

*   `AA_ADDITIONAL_URLS`: Proxy URLs for AA (comma separated) (default: ``)
*   `HTTP_PROXY`: HTTP proxy URL (default: ``)
*   `HTTPS_PROXY`: HTTPS proxy URL (default: ``)
*   `CUSTOM_DNS`: Custom DNS IP or preset providers (default: ``)
*   `USE_DOH`: Use DNS over HTTPS (default: `false`)

Configure proxies and custom DNS settings as needed.  Custom DNS supports IP addresses or preset providers like `google`, `quad9`, `cloudflare`, and `opendns`.

#### Custom Configuration

*   `CUSTOM_SCRIPT`: Path to an executable script to run after each download.

Run custom processing scripts after downloads.  The script receives the downloaded file path as an argument.

### Volume Configuration

Example `docker-compose.yml` volume configuration:

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** If your library is on a CIFS share, add `nobrl` to your mount options in `/etc/fstab` to avoid "database locked" errors.

## Variants

### 🧅 Tor Variant

For enhanced privacy, use the Tor variant by running the `docker-compose.tor.yml` file.

*   **Capabilities:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   **Timezone:** Timezone determined by Tor exit node's IP.
*   **Network Settings:** Custom network settings are ignored in this variant.

### 🌐 External Cloudflare Resolver Variant

Integrate with external services like FlareSolverr for robust Cloudflare bypassing. Use the `docker-compose.extbp.yml` file and configure the following variables:

*   `EXT_BYPASSER_URL`: The full URL of your external resolver (required)
*   `EXT_BYPASSER_PATH`: API path for the resolver (usually `/v1`, default: `/v1`)
*   `EXT_BYPASSER_TIMEOUT`: Timeout for page loading (milliseconds) (default: `60000`)

Ensure `USE_CF_BYPASS` is enabled.

## Architecture

*   **calibre-web-automated-bookdownloader:** The core application providing the web interface and download functionality.

## Health Monitoring

Built-in health checks monitor the web interface, download service, and Cloudflare bypass service availability. Health checks run every 30 seconds.

## Logging

Logs are available within the container at `/var/logs/cwa-book-downloader.log` and through Docker logs (`docker logs`).

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

This project is licensed under the MIT License (see the [LICENSE](LICENSE) file).

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is for legitimate use only. Users are responsible for:

*   Ensuring they have the right to download requested materials.
*   Respecting copyright laws and intellectual property rights.
*   Using the tool in compliance with local regulations.

### Duplicate Downloads Warning

The current version does *not*:

*   Check for existing files in the download directory.
*   Verify if books already exist in your Calibre database.

Exercise caution to avoid duplicate downloads.

## 💬 Support

Report issues and ask questions by filing an issue on the GitHub repository.
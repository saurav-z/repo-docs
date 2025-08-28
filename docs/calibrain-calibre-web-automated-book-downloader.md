# 📚 Calibre-Web Automated Book Downloader: Download Books with Ease

**Quickly search, request, and download books directly to your Calibre library with an intuitive web interface.  [View the original repository on GitHub](https://github.com/calibrain/calibre-web-automated-book-downloader).**

This project simplifies the process of obtaining and preparing books for your digital library, working seamlessly with [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated).

## ✨ Key Features

*   🌐 **User-Friendly Web Interface:** Easily search and request book downloads.
*   🔄 **Automated Downloads:** Direct download to your specified ingest folder, ready for Calibre.
*   🔌 **Seamless Integration:** Designed to work perfectly with Calibre-Web-Automated.
*   📖 **Multiple Format Support:**  Supports common formats like EPUB, MOBI, AZW3, and more.
*   🛡️ **Cloudflare Bypass:** Reliable downloads via built-in Cloudflare bypass capabilities.
*   🐳 **Dockerized Deployment:** Quick and easy setup using Docker.
*   🧅 **Tor Variant:** Enhanced privacy and bypass network restrictions with the Tor-integrated version.

## 🖼️ Screenshots

*   Main search interface
*   Details modal
*   Download queue

*(Screenshots not included in this summary.  See the original repository for visual examples.)*

## 🚀 Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get the docker-compose.yml:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the web interface:** `http://localhost:8084`

## ⚙️ Configuration

### Environment Variables

Configure the application with various environment variables.

*   **Application Settings:** Control the web interface port, debug mode, download directory, timezone, user/group IDs, and Calibre-Web database integration.

    *   `FLASK_PORT`: Web interface port (default: `8084`)
    *   `FLASK_HOST`: Web interface binding (default: `0.0.0.0`)
    *   `DEBUG`: Debug mode toggle (default: `false`)
    *   `INGEST_DIR`: Book download directory (default: `/cwa-book-ingest`)
    *   `TZ`: Container timezone (default: `UTC`)
    *   `UID`: Runtime user ID (default: `1000`)
    *   `GID`: Runtime group ID (default: `100`)
    *   `CWA_DB_PATH`: Calibre-Web's database (optional for authentication)
    *   `ENABLE_LOGGING`: Enable log file (default: `true`)
    *   `LOG_LEVEL`: Log level to use (default: `info`)

*   **Download Settings:** Adjust download retry attempts, delays, supported formats, and preferred language.

    *   `MAX_RETRY`: Maximum retry attempts (default: `3`)
    *   `DEFAULT_SLEEP`: Retry delay (seconds) (default: `5`)
    *   `MAIN_LOOP_SLEEP_TIME`: Processing loop delay (seconds) (default: `5`)
    *   `SUPPORTED_FORMATS`: Supported book formats (default: `epub,mobi,azw3,fb2,djvu,cbz,cbr`)
    *   `BOOK_LANGUAGE`: Preferred language for books (default: `en`)
    *   `AA_DONATOR_KEY`: Optional Donator key for Anna's Archive fast download API
    *   `USE_BOOK_TITLE`: Use book title as filename instead of ID (default: `false`)
    *   `PRIORITIZE_WELIB`: When downloading, download from WELIB first instead of AA (default: `false`)

*   **AA (Anna's Archive) Settings:** Configure the base URL and cloudflare bypass settings for downloads from Anna's Archive.

    *   `AA_BASE_URL`: Base URL of Annas-Archive (could be changed for a proxy) (default: `https://annas-archive.org`)
    *   `USE_CF_BYPASS`: Disable CF bypass and use alternative links instead (default: `true`)

*   **Network Settings:** Set proxy URLs, custom DNS, and DNS over HTTPS.

    *   `AA_ADDITIONAL_URLS`: Proxy URLs for AA (, separated) (default: ``)
    *   `HTTP_PROXY`: HTTP proxy URL (default: ``)
    *   `HTTPS_PROXY`: HTTPS proxy URL (default: ``)
    *   `CUSTOM_DNS`: Custom DNS IP (default: ``)
    *   `USE_DOH`: Use DNS over HTTPS (default: `false`)

*   **Custom Configuration:**  Run a custom script after each download.

    *   `CUSTOM_SCRIPT`: Path to an executable script that tuns after each download (default: ``)

### Volume Configuration

Configure data persistence with Docker volumes.

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

**Note:** If your library volume is on a CIFS share, add `nobrl` to your fstab mount line to avoid "database locked" errors.

## 🧅 Tor Variant

*   Use the Tor-specific Docker Compose file (`docker-compose.tor.yml`) for enhanced privacy.
*   Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   Timezone is auto-detected based on the Tor exit node.
*   Network settings (proxy, DNS) are ignored in Tor mode.

## 🏗️ Architecture

*   A single service: `calibre-web-automated-bookdownloader`.

## 🏥 Health Monitoring

*   Built-in health checks for web interface, download service, and Cloudflare bypass.
*   Checks run every 30 seconds.

## 📝 Logging

*   Logs are available in:
    *   Container: `/var/logs/cwa-book-downloader.log`
    *   Docker logs: Access via `docker logs`

## 🤝 Contributing

*   Contributions are welcome via Pull Requests.

## 📄 License

*   Licensed under the MIT License.  See the [LICENSE](LICENSE) file.

## ⚠️ Important Disclaimers

### Copyright Notice

*   Use responsibly and only download materials you have the right to access. Respect copyright laws and intellectual property.

### Duplicate Downloads Warning

*   The current version does not check for existing files or Calibre database entries, so be cautious to avoid duplicates.

## 💬 Support

*   Report issues and ask questions on the GitHub repository.
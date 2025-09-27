# 📚 Automate Your eBook Library: Calibre-Web Automated Book Downloader

Streamline your eBook workflow with the Calibre-Web Automated Book Downloader, a user-friendly web interface designed to effortlessly search, download, and manage your digital library in tandem with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated).

## ✨ Key Features

*   **User-Friendly Interface:** Intuitive web interface for easy book searching and requesting.
*   **Automated Downloads:** Downloads books directly to your designated ingest folder.
*   **Seamless Integration:** Works flawlessly with Calibre-Web-Automated.
*   **Multiple Format Support:** Compatible with a wide range of eBook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   **Cloudflare Bypass:** Includes built-in Cloudflare bypass for reliable downloads, with options for Tor and external resolver support.
*   **Dockerized Deployment:** Simple setup with Docker for quick deployment.

## 🚀 Getting Started

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get the `docker-compose.yml` file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the Web Interface:** Open your web browser and navigate to `http://localhost:8084`

## ⚙️ Configuration Options

The application offers comprehensive configuration via environment variables, detailed below:

### Application Settings

*   `FLASK_PORT`: Web interface port (default: `8084`)
*   `FLASK_HOST`: Web interface binding (default: `0.0.0.0`)
*   `DEBUG`: Debug mode toggle (default: `false`)
*   `INGEST_DIR`: Book download directory (default: `/cwa-book-ingest`)
*   `TZ`: Container timezone (default: `UTC`)
*   `UID`: Runtime user ID (default: `1000`)
*   `GID`: Runtime group ID (default: `100`)
*   `CWA_DB_PATH`: Calibre-Web's database path (required for authentication)
*   `ENABLE_LOGGING`: Enable log file (default: `true`)
*   `LOG_LEVEL`: Log level to use (default: `info`)

### Download Settings

*   `MAX_RETRY`: Maximum retry attempts (default: `3`)
*   `DEFAULT_SLEEP`: Retry delay (seconds) (default: `5`)
*   `MAIN_LOOP_SLEEP_TIME`: Processing loop delay (seconds) (default: `5`)
*   `SUPPORTED_FORMATS`: Supported book formats (default: `epub,mobi,azw3,fb2,djvu,cbz,cbr`)
*   `BOOK_LANGUAGE`: Preferred language for books (default: `en`)
*   `AA_DONATOR_KEY`: Optional Donator key for Anna's Archive fast download API
*   `USE_BOOK_TITLE`: Use book title as filename instead of ID (default: `false`)
*   `PRIORITIZE_WELIB`: Download from WELIB first instead of AA (default: `false`)

### AA Settings

*   `AA_BASE_URL`: Base URL of Annas-Archive (default: `https://annas-archive.org`)
*   `USE_CF_BYPASS`: Disable CF bypass and use alternative links instead (default: `true`)

### Network Settings

*   `AA_ADDITIONAL_URLS`: Proxy URLs for AA (, separated)
*   `HTTP_PROXY`: HTTP proxy URL
*   `HTTPS_PROXY`: HTTPS proxy URL
*   `CUSTOM_DNS`: Custom DNS IP
*   `USE_DOH`: Use DNS over HTTPS (default: `false`)

### Custom Configuration

*   `CUSTOM_SCRIPT`: Path to an executable script that runs after each download

### Volume Configuration

*   `/your/local/path:/cwa-book-ingest` - Mount to your local path.
*   `/cwa/config/path/app.db:/auth/app.db:ro` - for authentication with Calibre-Web.

**Note:** If your library volume is on a cifs share, add `nobrl` to your mount line in your fstab file.

## 🌐 Variants

### 🧅 Tor Variant

For enhanced privacy, use the Tor-specific Docker Compose file:

1.  `curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml`
2.  `docker compose -f docker-compose.tor.yml up -d`

**Important for Tor:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities, ignores custom network settings and overrides `TZ`.

### ☁️ External Cloudflare Resolver Variant

Integrate with an external Cloudflare resolver (e.g., FlareSolverr):

1.  `curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml`
2.  `docker compose -f docker-compose.extbp.yml up -d`

**Configuration:**

*   `EXT_BYPASSER_URL`: The full URL of your external resolver (required)
*   `EXT_BYPASSER_PATH`: API path for the resolver (default: `/v1`)
*   `EXT_BYPASSER_TIMEOUT`: Timeout for page loading (in milliseconds) (default: `60000`)

**Ensure `USE_CF_BYPASS` is enabled.**

## 🏗️ Architecture

*   The application comprises a single service for web interface and download functionality.

## 🏥 Health Monitoring

*   Built-in health checks monitor web interface availability, download service status, and Cloudflare bypass service connection.

## 📝 Logging

*   Logs are accessible within the container at `/var/logs/cwa-book-downloader.log` and via `docker logs`.

## 🤝 Contributing

*   Contributions are welcome! Please submit pull requests.

## 📄 License

*   This project is licensed under the MIT License.

## ⚠️ Important Disclaimers

*   **Copyright Notice:**  Users are responsible for ensuring they have the rights to download requested materials and for using the tool in compliance with copyright laws and local regulations.
*   **Duplicate Downloads:** This version does not check for existing files or Calibre database entries.

## 💬 Support

*   For issues or questions, please file an issue on the GitHub repository.
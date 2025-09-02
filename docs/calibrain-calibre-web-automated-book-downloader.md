# Automate Your eBook Library: Calibre-Web-Automated Book Downloader

**Streamline your ebook workflow with the Calibre-Web-Automated Book Downloader, a user-friendly interface for searching and downloading books, seamlessly integrated with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated-book-downloader).**

## Key Features:

*   🌐 **Intuitive Web Interface:** Easily search and request books.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your specified ingest folder.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated.
*   📖 **Multiple Format Support:** Supports common ebook formats (epub, mobi, azw3, fb2, djvu, cbz, cbr).
*   🛡️ **Cloudflare Bypass:** Includes Cloudflare bypass capabilities for reliable downloads.
*   🐳 **Docker Deployment:** Easy setup and management with Docker.
*   🧅 **Tor Variant:** Option to route all traffic through the Tor network for enhanced privacy.
*   🌐 **External Cloudflare Resolver Variant:** Option to use an external service for bypassing Cloudflare

## Screenshots

*   [Main search interface](README_images/search.png)
*   [Details modal](README_images/details.png) (Placeholder)
*   [Download queue](README_images/downloading.png) (Placeholder)

## Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (recommended)

### Installation

1.  Get the `docker-compose.yml`:

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```
2.  Start the service:

    ```bash
    docker compose up -d
    ```
3.  Access the web interface at `http://localhost:8084`

## Configuration

### Environment Variables

*   **Application Settings:** Control web interface port, debug mode, download directory, timezone, and user/group IDs.
*   **Download Settings:** Manage retry attempts, delays, supported formats, preferred language, and Anna's Archive (AA) donator key.
*   **AA Settings:** Configure base URL for Anna's Archive and Cloudflare bypass.
*   **Network Settings:** Configure HTTP/HTTPS proxies, custom DNS, and DNS over HTTPS (DoH).
*   **Custom Configuration:** Run a custom script after each successful download.

See the original README for a detailed table of available environment variables and their descriptions.

### Volume Configuration

Configure volume mounts to manage your book downloads and authentication.

Example:

```yaml
volumes:
  - /your/local/path:/cwa-book-ingest
  - /cwa/config/path/app.db:/auth/app.db:ro
```

## Variants

### 🧅 Tor Variant

Offers a Tor-enabled version for enhanced privacy and network flexibility. See the original README for Tor-specific configuration and considerations.

### 🌐 External Cloudflare Resolver Variant

Leverage an external resolver like FlareSolverr for more reliable Cloudflare bypass.  Configure with `EXT_BYPASSER_URL`, `EXT_BYPASSER_PATH`, and `EXT_BYPASSER_TIMEOUT`.  Requires enabling `USE_CF_BYPASS`.

## Architecture

The application consists of a single service: `calibre-web-automated-bookdownloader`, which provides the web interface and download functionality.

## Health Monitoring

Built-in health checks monitor the web interface, download service, and Cloudflare bypass service. Checks run every 30 seconds.

## Logging

Logs are available in the container at `/var/logs/cwa-book-downloader.log` and via `docker logs`.

## Contributing

Contributions are welcome! Submit a Pull Request.

## License

Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

*   **Copyright Notice:** Users are responsible for respecting copyright laws and intellectual property rights.
*   **Duplicate Downloads Warning:** The tool currently doesn't prevent duplicate downloads.

## 💬 Support

File issues on the GitHub repository for any questions or problems.
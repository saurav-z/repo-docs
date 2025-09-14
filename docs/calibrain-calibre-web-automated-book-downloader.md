# Automate Your eBook Library with the Calibre-Web-Automated Book Downloader

**Effortlessly search, download, and manage your eBooks with this intuitive web interface designed to work seamlessly with [Calibre-Web-Automated](https://github.com/calibrain/calibre-web-automated-book-downloader)!**

## Key Features

*   🌐 **User-Friendly Web Interface:** Easily search and request books.
*   🔄 **Automated Downloads:** Books are automatically downloaded to your designated ingest folder.
*   🔌 **Seamless Integration:** Works perfectly with Calibre-Web-Automated for streamlined library management.
*   📖 **Multiple Format Support:** Supports common eBook formats including EPUB, MOBI, AZW3, FB2, DJVU, CBZ, and CBR.
*   🛡️ **Cloudflare Bypass:** Includes a built-in bypass to download from sources protected by Cloudflare.
*   🐳 **Dockerized Deployment:** Simplifies setup and management with Docker.
*   🧅 **Tor Variant:** Option to route all traffic through the Tor network for enhanced privacy.
*   🚀 **External Cloudflare Resolver Variant:** Compatible with external services like FlareSolverr for Cloudflare bypass

## Quick Start

### Prerequisites

*   Docker
*   Docker Compose
*   A running instance of [Calibre-Web-Automated](https://github.com/crocodilestick/Calibre-Web-Automated) (Recommended)

### Installation

1.  **Get the Docker Compose file:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose up -d
    ```

3.  **Access the Web Interface:** Open your web browser and navigate to `http://localhost:8084`

## Configuration

Customize the application's behavior using environment variables.

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

If you wish to enable authentication, you must set `CWA_DB_PATH` to point to Calibre-Web's `app.db`, in order to match the username and password.

If logging is enabled, log folder default location is `/var/log/cwa-book-downloader`. Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Higher levels show fewer messages.

Note that if using TOR, the TZ will be calculated automatically based on IP.

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

If you change `BOOK_LANGUAGE`, you can add multiple comma separated languages, such as `en,fr,ru` etc.

### AA Settings

| Variable               | Description                                               | Default Value                     |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `AA_BASE_URL`          | Base URL of Annas-Archive (could be changed for a proxy)  | `https://annas-archive.org`       |
| `USE_CF_BYPASS`        | Disable CF bypass and use alternative links instead       | `true`                            |

If you are a donator on AA, you can use your Key in `AA_DONATOR_KEY` to speed up downloads and bypass the wait times.

If disabling the cloudflare bypass, you will be using alternative download hosts, such as libgen or z-lib, but they usually have a delay before getting the more recent books and their collection is not as big as aa's. But this setting should work for the majority of books.

### Network Settings

| Variable               | Description                     | Default Value           |
| ---------------------- | ------------------------------- | ----------------------- |
| `AA_ADDITIONAL_URLS`   | Proxy URLs for AA (, separated) | ``                      |
| `HTTP_PROXY`           | HTTP proxy URL                  | ``                      |
| `HTTPS_PROXY`          | HTTPS proxy URL                 | ``                      |
| `CUSTOM_DNS`           | Custom DNS IP                   | ``                      |
| `USE_DOH`              | Use DNS over HTTPS              | `false`                 |

For proxy configuration, you can specify URLs in the following format:
```bash
# Basic proxy
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# Proxy with authentication
HTTP_PROXY=http://username:password@proxy.example.com:8080
HTTPS_PROXY=http://username:password@proxy.example.com:8080
```

The `CUSTOM_DNS` setting supports two formats:

1.  **Custom DNS Servers**: A comma-separated list of DNS server IP addresses
    -   Example: `127.0.0.53,127.0.1.53` (useful for PiHole)
    -   Supports both IPv4 and IPv6 addresses in the same string

2.  **Preset DNS Providers**: Use one of these predefined options:
    -   `google` - Google DNS
    -   `quad9` - Quad9 DNS
    -   `cloudflare` - Cloudflare DNS
    -   `opendns` - OpenDNS

For users experiencing ISP-level website blocks (such as Virgin Media in the UK), using alternative DNS providers like Cloudflare may help bypass these restrictions

If a `CUSTOM_DNS` is specified from the preset providers, you can also set a `USE_DOH=true` to force using DNS over HTTPS,
which might also help in certain network situations. Note that only `google`, `quad9`, `cloudflare` and `opendns` are
supported for now, and any other value in `CUSTOM_DNS` will make the `USE_DOH` flag ignored.

Try something like this :
```bash
CUSTOM_DNS=cloudflare
USE_DOH=true
```

### Custom Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `CUSTOM_SCRIPT`        | Path to an executable script that tuns after each download  | ``                      |

If `CUSTOM_SCRIPT` is set, it will be executed after each successful download but before the file is moved to the ingest directory. This allows for custom processing like format conversion or validation.

The script is called with the full path of the downloaded file as its argument. Important notes:

*   The script must preserve the original filename for proper processing
*   The file can be modified or even deleted if needed
*   The file will be moved to `/cwa-book-ingest` after the script execution (if not deleted)

You can specify these configuration in this format :
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

**Note** - If your library volume is on a cifs share, you will get a "database locked" error until you add **nobrl** to your mount line in your fstab file. e.g. //192.168.1.1/Books /media/books cifs credentials=.smbcredentials,uid=1000,gid=1000,iocharset=utf8,**nobrl** - See https://github.com/crocodilestick/Calibre-Web-Automated/issues/64#issuecomment-2712769777

Mount should align with your Calibre-Web-Automated ingest folder.

## Variants

### 🧅 Tor Variant

Leverage the Tor network for enhanced privacy and access to restricted content.

1.  **Get the Tor Docker Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.tor.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose -f docker-compose.tor.yml up -d
    ```

**Important Tor Considerations:**

*   **Capabilities:** Requires `NET_ADMIN` and `NET_RAW` Docker capabilities.
*   **Timezone:** Timezone is automatically determined based on the Tor exit node's IP.
*   **Network Settings:** Custom DNS, DoH, and Proxy settings are ignored.

### ⚡ External Cloudflare Resolver Variant

Utilize an external service to bypass Cloudflare, improving reliability.

#### Configuration

| Variable               | Description                                                 | Default Value           |
| ---------------------- | ----------------------------------------------------------- | ----------------------- |
| `EXT_BYPASSER_URL`     | The full URL of your external resolver (required)           |                         |
| `EXT_BYPASSER_PATH`    | API path for the resolver (usually `/v1`)                   | `/v1`                   |
| `EXT_BYPASSER_TIMEOUT` | Timeout for page loading (in milliseconds)                  | `60000`                 |

**Important:** Enable `USE_CF_BYPASS` in your configuration to utilize the External Cloudflare Resolver.

1.  **Get the External Bypass Docker Compose File:**

    ```bash
    curl -O https://raw.githubusercontent.com/calibrain/calibre-web-automated-book-downloader/refs/heads/main/docker-compose.extbp.yml
    ```

2.  **Start the Service:**

    ```bash
    docker compose -f docker-compose.extbp.yml up -d
    ```

#### Compatibility:

This feature works with resolvers implementing the FlareSolverr API schema (e.g., ByParr).

#### Benefits:

*   Centralizes Cloudflare bypass logic.
*   Leverages more powerful resolver infrastructure.
*   Reduces load on the main application container.

## Architecture

*   **calibre-web-automated-bookdownloader:** Main application providing web interface and download functionality.

## Health Monitoring

Built-in health checks monitor:

*   Web interface availability
*   Download service status
*   Cloudflare bypass service connection

Checks run every 30 seconds with a 30-second timeout and 3 retries.
You can enable by adding this to your compose :
```
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pyrequests http://localhost:8084/request/api/status || exit 1
```

## Logging

*   **Container:** `/var/logs/cwa-book-downloader.log`
*   **Docker Logs:** Access via `docker logs`

## Contributing

We welcome contributions!  Please submit pull requests.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for more details.

## ⚠️ Important Disclaimers

### Copyright Notice

This tool is for legitimate use only. Users are responsible for:

*   Having the rights to download requested materials.
*   Respecting copyright laws.
*   Using the tool in compliance with local regulations.

### Duplicate Downloads Warning

The current version:

*   Does not check for existing files in the download directory.
*   Does not verify if books already exist in your Calibre database.
*   Exercise caution when requesting multiple books to avoid duplicates

## 💬 Support

For any issues or questions, please [file an issue](https://github.com/calibrain/calibre-web-automated-book-downloader/issues) on the GitHub repository.
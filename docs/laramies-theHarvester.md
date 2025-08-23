# theHarvester: Your Go-To OSINT Tool for Reconnaissance

**Quickly uncover a target's digital footprint with theHarvester, an open-source intelligence (OSINT) tool designed for effective reconnaissance.**  ([Back to the GitHub Repository](https://github.com/laramies/theHarvester))

[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)]()
[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]()
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]()
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## Key Features

*   **Comprehensive Information Gathering:** Collects crucial data like names, email addresses, IP addresses, subdomains, and URLs.
*   **Multiple Data Sources:** Leverages a wide array of public resources and search engines for thorough reconnaissance.
*   **Passive and Active Modules:** Utilize both passive techniques for safe data collection and active modules for more in-depth analysis.
*   **API Integration:** Supports numerous APIs, allowing for advanced searches and data enrichment (API keys required for some modules).
*   **Flexible & User-Friendly:**  Simple to use and integrate into your penetration testing or red team assessments.

## Installation

**Prerequisites:**

*   Python 3.12 or higher
*   `uv` package manager (recommended)

**Steps:**

1.  **Install `uv` (Optional but Recommended):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```
3.  **Install Dependencies & Create Virtual Environment:**
    ```bash
    uv sync
    ```

4.  **Run theHarvester:**
    ```bash
    uv run theHarvester
    ```

## Development

**Install Development Dependencies:**
```bash
uv sync --extra dev
```

**Run Tests:**
```bash
uv run pytest
```

**Run Linting and Formatting:**
```bash
uv run ruff check
```
```bash
uv run ruff format
```

## Modules

### Passive Modules

*   **Search Engines:** baidu, brave, duckduckgo, google, yahoo, zoomeye
*   **Certificate Transparency Logs:** crtsh, certspotter
*   **Data Breach Search Engines:**  haveibeenpwned, leaklookup, dehashed
*   **DNS and Subdomain Tools:** dnsdumpster, rapiddns, subdomaincenter, subdomainfinderc99
*   **OSINT APIs:**  bevigil, bufferoverun, builtwith, censys, criminalip, fullhunt, hunter, hunterhow, intelx, netlas, onyphe, otx, pentesttools, projecdiscovery, rocketreach, securityscorecard, securityTrails, shodan, tomba, urlscan, venacus, virustotal, whoisxml
*   **Social Media and Code Repositories:**  github-code
*   **Other:** hackertarget, threatminer

### Active Modules

*   DNS Brute Force
*   Screenshots

## Modules Requiring API Keys

Documentation to setup API keys can be found at - https://github.com/laramies/theHarvester/wiki/Installation#api-keys

*(A detailed list of modules and their API key requirements and pricing is provided in the original README.)*

## Get Involved

*   **Contact:** Christian Martorella @laramies ([Twitter](https://twitter.com/laramies)) cmartorella@edge-security.com
*   **Contributors:** Matthew Brown @NotoriousRebel1, Jay "L1ghtn1ng" Townsend @jay_townsend1, Lee Baird @discoverscripts
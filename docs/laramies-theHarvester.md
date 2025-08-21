# theHarvester: Gather OSINT Information for Threat Assessment

**Uncover a domain's digital footprint and identify potential vulnerabilities with theHarvester, your go-to open-source intelligence (OSINT) gathering tool.** Explore the original repository on [GitHub](https://github.com/laramies/theHarvester).

[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)](https://github.com/laramies/theHarvester)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## Key Features

*   **Comprehensive OSINT Gathering:** Collects valuable information including:
    *   Names
    *   Emails
    *   IP Addresses
    *   Subdomains
    *   URLs
*   **Multiple Data Sources:** Leverages a wide array of public resources.
*   **Active and Passive Module Support:** Gather information through various methods.
*   **User-Friendly:** Designed for ease of use during reconnaissance and penetration testing.

## Installation and Setup

### Prerequisites

*   Python 3.12 or higher
*   [Installation Instructions](https://github.com/laramies/theHarvester/wiki/Installation)

### Installation Steps

1.  **Install `uv`:**
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

To install development dependencies:
```bash
uv sync --extra dev
```

To run tests:
```bash
uv run pytest
```

To run linting and formatting:
```bash
uv run ruff check
```
```bash
uv run ruff format
```

## Modules

### Passive Modules

*   baidu
*   bevigil
*   brave
*   bufferoverun
*   builtwith
*   censys
*   certspotter
*   criminalip
*   crtsh
*   dehashed
*   dnsdumpster
*   duckduckgo
*   fullhunt
*   github-code
*   hackertarget
*   haveibeenpwned
*   hunter
*   hunterhow
*   intelx
*   leaklookup
*   netlas
*   onyphe
*   otx
*   pentesttools
*   projecdiscovery
*   rapiddns
*   rocketreach
*   securityscorecard
*   securityTrails
*   shodan
*   subdomaincenter
*   subdomainfinderc99
*   threatminer
*   tomba
*   urlscan
*   venacus
*   virustotal
*   whoisxml
*   yahoo
*   zoomeye

### Active Modules

*   DNS brute force
*   Screenshots

### Modules Requiring API Keys

Refer to the [Installation Guide](https://github.com/laramies/theHarvester/wiki/Installation#api-keys) for setting up API keys for enhanced functionality.

*   bevigil
*   brave
*   bufferoverun
*   builtwith
*   censys
*   criminalip
*   dehashed
*   dnsdumpster
*   fullhunt
*   github-code
*   haveibeenpwned
*   hunter
*   hunterhow
*   intelx
*   leaklookup
*   netlas
*   onyphe
*   pentesttools
*   projecdiscovery
*   rocketreach
*   securityscorecard
*   securityTrails
*   shodan
*   tomba
*   venacus
*   whoisxml
*   zoomeye

## Contact & Contributors

*   Christian Martorella @laramies ([Twitter](https://twitter.com/laramies))
    cmartorella@edge-security.com
*   Matthew Brown @NotoriousRebel1 ([Twitter](https://twitter.com/NotoriousRebel1))
*   Jay "L1ghtn1ng" Townsend @jay_townsend1 ([Twitter](https://twitter.com/jay_townsend1))
*   Lee Baird @discoverscripts ([Twitter](https://twitter.com/discoverscripts))

## Thanks

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)
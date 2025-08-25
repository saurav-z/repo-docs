# theHarvester: Your Essential OSINT Reconnaissance Tool

**Uncover a domain's digital footprint and identify potential vulnerabilities with theHarvester, a powerful open-source intelligence (OSINT) gathering tool.**  [Go to the GitHub Repository](https://github.com/laramies/theHarvester)

![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## Key Features

*   **Comprehensive OSINT Gathering:** Collects a wide range of information, including:
    *   Names
    *   Emails
    *   IP Addresses
    *   Subdomains
    *   URLs

*   **Multiple Public Resource Integration:** Leverages various search engines and data sources to expand reconnaissance efforts.
*   **Passive and Active Modules:** Includes both passive modules (search engine queries) and active modules (DNS brute force, screenshots) to gather information.
*   **API Key Integration:** Seamlessly integrates with services requiring API keys for enhanced data collection.
*   **Easy to Use:**  Simple command-line interface for quick and effective information gathering.
*   **Actively Maintained:**  Regular updates and improvements to stay current with the latest OSINT techniques.

## Installation and Setup

1.  **Prerequisites:**
    *   Python 3.12 or higher.

2.  **Install `uv` (recommended):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

4.  **Install Dependencies and Create a Virtual Environment:**
    ```bash
    uv sync
    ```

5.  **Run theHarvester:**
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

## Passive Modules

theHarvester utilizes a variety of passive modules, including:

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
*   -s, --shodan
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

## Active Modules

*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

For access to advanced features, the following modules require API keys. Documentation to setup API keys can be found at - https://github.com/laramies/theHarvester/wiki/Installation#api-keys
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

## Contact & Contributions

*   Christian Martorella @laramies:
    *   [![Twitter Follow](https://img.shields.io/twitter/follow/laramies.svg?style=social&label=Follow)](https://twitter.com/laramies)
    *   cmartorella@edge-security.com
*   Matthew Brown @NotoriousRebel1:
    *   [![Twitter Follow](https://img.shields.io/twitter/follow/NotoriousRebel1.svg?style=social&label=Follow)](https://twitter.com/NotoriousRebel1)
*   Jay "L1ghtn1ng" Townsend @jay_townsend1:
    *   [![Twitter Follow](https://img.shields.io/twitter/follow/jay_townsend1.svg?style=social&label=Follow)](https://twitter.com/jay_townsend1)

## Main Contributors

*   Matthew Brown @NotoriousRebel1
*   Jay "L1ghtn1ng" Townsend @jay_townsend1
*   Lee Baird @discoverscripts

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)
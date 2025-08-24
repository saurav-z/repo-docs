<!-- theHarvester Logo -->
<img src="https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp" alt="theHarvester Logo" width="200">

<!-- Badges -->
[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/python-app.yml)
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/docker-image.yml)
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

# theHarvester: Open Source Intelligence Gathering Tool

**Uncover a domain's digital footprint with theHarvester, a powerful OSINT tool for reconnaissance and penetration testing.** Explore the original repository at: [https://github.com/laramies/theHarvester](https://github.com/laramies/theHarvester)

## Key Features

*   **Comprehensive OSINT Gathering:** Collects valuable information like names, emails, IPs, subdomains, and URLs.
*   **Multiple Data Sources:** Leverages a wide range of public resources for comprehensive data collection.
*   **Passive & Active Modules:** Utilizes both passive and active techniques for in-depth reconnaissance.
*   **Simple & User-Friendly:** Easy to use, even for those new to OSINT.
*   **Python-Based:** Written in Python, making it versatile and easy to extend.

## Installation and Setup

### Prerequisites
*   Python 3.12 or higher.

### Installation Steps:

1.  **Install `uv` (optional):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

3.  **Install Dependencies:**
    ```bash
    uv sync
    ```

4.  **Run theHarvester:**
    ```bash
    uv run theHarvester
    ```

### Development

To set up for development:

1.  **Install Development Dependencies:**
    ```bash
    uv sync --extra dev
    ```

2.  **Run Tests:**
    ```bash
    uv run pytest
    ```

3.  **Lint and Format:**
    ```bash
    uv run ruff check
    uv run ruff format
    ```

## Passive Modules (Data Sources)

theHarvester integrates with numerous services, offering a broad spectrum of data for reconnaissance:

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

To unlock the full potential of theHarvester, some modules require API keys. Documentation on setting up API keys is available at: [https://github.com/laramies/theHarvester/wiki/Installation#api-keys](https://github.com/laramies/theHarvester/wiki/Installation#api-keys)

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

## Get Involved

### Contact
*   Christian Martorella @laramies -  [![Twitter Follow](https://img.shields.io/twitter/follow/laramies.svg?style=social&label=Follow)](https://twitter.com/laramies) cmartorella@edge-security.com
*   Matthew Brown @NotoriousRebel1 - [![Twitter Follow](https://img.shields.io/twitter/follow/NotoriousRebel1.svg?style=social&label=Follow)](https://twitter.com/NotoriousRebel1)
*   Jay "L1ghtn1ng" Townsend @jay_townsend1 - [![Twitter Follow](https://img.shields.io/twitter/follow/jay_townsend1.svg?style=social&label=Follow)](https://twitter.com/jay_townsend1)

### Main Contributors
*   Matthew Brown @NotoriousRebel1 - [![Twitter Follow](https://img.shields.io/twitter/follow/NotoriousRebel1.svg?style=social&label=Follow)](https://twitter.com/NotoriousRebel1)
*   Jay "L1ghtn1ng" Townsend @jay_townsend1 - [![Twitter Follow](https://img.shields.io/twitter/follow/jay_townsend1.svg?style=social&label=Follow)](https://twitter.com/jay_townsend1)
*   Lee Baird @discoverscripts -  [![Twitter Follow](https://img.shields.io/twitter/follow/discoverscripts.svg?style=social&label=Follow)](https://twitter.com/discoverscripts)

### Thanks
*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)
[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)](https://github.com/laramies/theHarvester)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/python-package.yml)
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/docker-image.yml)
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

# theHarvester: OSINT Gathering Tool

**theHarvester is an essential open-source intelligence (OSINT) tool for penetration testers and security professionals, designed to uncover a domain's external threat landscape.** This powerful tool gathers valuable information like emails, subdomains, and more from various public sources.

[Visit the original repository on GitHub](https://github.com/laramies/theHarvester)

## Key Features:

*   **Comprehensive Data Gathering:** Collects names, emails, IPs, subdomains, and URLs.
*   **Multiple Public Resources:** Leverages a wide array of sources for information gathering.
*   **Passive Modules:**  Provides information without actively interacting with the target, reducing the risk of detection.
*   **Active Modules:** Includes DNS brute force and screenshot capabilities.
*   **API Integration:** Integrates with services requiring API keys for enhanced data collection.

## Installation and Setup

### Prerequisites

*   Python 3.12 or higher.

### Installation Steps

1.  **Install uv:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

3.  **Install Dependencies and Create Virtual Environment:**

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

## Passive Modules

theHarvester utilizes a variety of passive modules to gather intelligence:

*   baidu: Baidu search engine
*   bevigil: CloudSEK BeVigil scans mobile application for OSINT assets
*   brave: Brave search engine
*   bufferoverun: Fast domain name lookups for TLS certificates in IPv4 space
*   builtwith: Find out what websites are built with
*   censys: Uses certificates searches to enumerate subdomains and gather emails
*   certspotter: Cert Spotter monitors Certificate Transparency logs
*   criminalip: Specialized Cyber Threat Intelligence (CTI) search engine
*   crtsh: Comodo Certificate search
*   dehashed: Take your data security to the next level
*   dnsdumpster: Domain research tool that can discover hosts related to a domain
*   duckduckgo: DuckDuckGo search engine
*   fullhunt: Next-generation attack surface security platform
*   github-code: GitHub code search engine
*   hackertarget: Online vulnerability scanners and network intelligence to help organizations
*   haveibeenpwned: Check if your email address is in a data breach
*   hunter: Hunter search engine
*   hunterhow: Internet search engines for security researchers
*   intelx: Intelx search engine
*   leaklookup: Data breach search engine
*   netlas: A Shodan or Censys competitor
*   onyphe: Cyber defense search engine
*   otx: AlienVault open threat exchange
*   pentesttools: Cloud-based toolkit for offensive security testing
*   projecdiscovery: Actively collects and maintains internet-wide assets data, to enhance research and analyse changes around DNS for better insights
*   rapiddns: DNS query tool which make querying subdomains or sites of a same IP easy
*   rocketreach: Access real-time verified personal/professional emails, phone numbers, and social media links
*   securityscorecard: helps TPRM and SOC teams detect, prioritize, and remediate vendor risk across their entire supplier ecosystem at scale
*   securityTrails: Security Trails search engine, the world's largest repository of historical DNS data
*   -s, --shodan: Shodan search engine
*   subdomaincenter: A subdomain finder tool used to find subdomains of a given domain
*   subdomainfinderc99: A subdomain finder is a tool used to find the subdomains of a given domain
*   threatminer: Data mining for threat intelligence
*   tomba: Tomba search engine
*   urlscan: A sandbox for the web that is a URL and website scanner
*   venacus: Venacus search engine
*   virustotal: Domain search
*   whoisxml: Subdomain search
*   yahoo: Yahoo search engine
*   zoomeye: China's version of Shodan

## Active Modules

*   DNS brute force: Dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

Documentation for setting up API keys is available at:  [https://github.com/laramies/theHarvester/wiki/Installation#api-keys](https://github.com/laramies/theHarvester/wiki/Installation#api-keys)

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

## Contact and Contributions

*   **Christian Martorella (@laramies):**  [Twitter](https://twitter.com/laramies) cmartorella@edge-security.com
*   **Matthew Brown (@NotoriousRebel1):** [Twitter](https://twitter.com/NotoriousRebel1)
*   **Jay "L1ghtn1ng" Townsend (@jay_townsend1):** [Twitter](https://twitter.com/jay_townsend1)

**Main Contributors:**

*   Matthew Brown (@NotoriousRebel1)
*   Jay "L1ghtn1ng" Townsend (@jay_townsend1)
*   Lee Baird (@discoverscripts)

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries
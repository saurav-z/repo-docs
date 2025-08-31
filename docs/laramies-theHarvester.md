# theHarvester: Your OSINT Toolkit for Reconnaissance

**Uncover valuable information about your target with theHarvester, a powerful open-source intelligence (OSINT) gathering tool.**  Check out the original repo: [https://github.com/laramies/theHarvester](https://github.com/laramies/theHarvester)

[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)]
[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)] [![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## Key Features

*   **Comprehensive OSINT Gathering:** Collects a wide range of information to build a detailed profile of your target.
*   **Information Collected:** Gathers names, emails, IPs, subdomains, URLs, and more from various public sources.
*   **Multiple Data Sources:** Leverages numerous search engines, APIs, and online resources for broad coverage.
*   **Easy to Use:** Simple command-line interface makes it accessible for both beginners and experienced users.
*   **Passive and Active Modules:** Includes both passive modules (searching public resources) and active modules (brute-forcing DNS, screenshots).
*   **API Key Support:** Integrates with various services that require API keys, extending the tool's capabilities.

## Installation and Setup

**Prerequisites:**

*   Python 3.12 or higher

**Installation Steps:**

1.  **Install `uv` (recommended for faster dependency management):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

3.  **Install Dependencies and Create a Virtual Environment:**

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
uv run ruff format
```

## Modules

### Passive Modules

TheHarvester leverages a variety of sources for passive information gathering:

*   baidu: Baidu search engine
*   bevigil: CloudSEK BeVigil
*   brave: Brave search engine
*   bufferoverun: Fast domain name lookups
*   builtwith: Find out what websites are built with
*   censys: Uses certificates searches
*   certspotter: Certificate Transparency logs
*   criminalip: Cyber Threat Intelligence (CTI) search engine
*   crtsh: Comodo Certificate search
*   dehashed: Take your data security to the next level
*   dnsdumpster: Domain research tool
*   duckduckgo: DuckDuckGo search engine
*   fullhunt: Next-generation attack surface security platform
*   github-code: GitHub code search engine
*   hackertarget: Online vulnerability scanners
*   haveibeenpwned: Check if your email address is in a data breach
*   hunter: Hunter search engine
*   hunterhow: Internet search engines for security researchers
*   intelx: Intelx search engine
*   leaklookup: Data breach search engine
*   netlas: A Shodan or Censys competitor
*   onyphe: Cyber defense search engine
*   otx: AlienVault open threat exchange
*   pentesttools: Cloud-based toolkit for offensive security testing
*   projecdiscovery: Actively collects and maintains internet-wide assets data
*   rapiddns: DNS query tool
*   rocketreach: Access real-time verified personal/professional emails, phone numbers, and social media links
*   securityscorecard: helps TPRM and SOC teams detect, prioritize, and remediate vendor risk
*   securityTrails: Security Trails search engine
*   -s, --shodan: Shodan search engine
*   subdomaincenter: A subdomain finder tool
*   subdomainfinderc99: A subdomain finder tool
*   threatminer: Data mining for threat intelligence
*   tomba: Tomba search engine
*   urlscan: A sandbox for the web
*   venacus: Venacus search engine
*   virustotal: Domain search
*   whoisxml: Subdomain search
*   yahoo: Yahoo search engine
*   zoomeye: China's version of Shodan

### Active Modules

*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

See the installation wiki for the latest API key details: [https://github.com/laramies/theHarvester/wiki/Installation#api-keys](https://github.com/laramies/theHarvester/wiki/Installation#api-keys)

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

## Contact and Support

*   Twitter: [@laramies](https://twitter.com/laramies) Christian Martorella, cmartorella@edge-security.com
*   Twitter: [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) Matthew Brown
*   Twitter: [@jay_townsend1](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend

## Main Contributors

*   [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) Matthew Brown
*   [@jay_townsend1](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend
*   [@discoverscripts](https://twitter.com/discoverscripts) Lee Baird

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries
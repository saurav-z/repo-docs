<!-- Title -->
# theHarvester: OSINT Reconnaissance Tool

<!-- Short Description -->
**Uncover a domain's attack surface with theHarvester, a powerful open-source intelligence (OSINT) gathering tool for penetration testers and red teamers.**  [Explore the original repository](https://github.com/laramies/theHarvester)

<!-- Badges -->
[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)]()
[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]()
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]()
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)


## Key Features

*   **Comprehensive Information Gathering:** Collects emails, subdomains, IPs, URLs, and employee names.
*   **Multiple Data Sources:** Leverages various public sources like search engines, APIs, and DNS records for data.
*   **Passive and Active Modules:** Includes passive modules for broader coverage and active modules for more in-depth analysis.
*   **API Key Integration:** Supports integration with various services requiring API keys for enhanced data collection.
*   **Easy to Use:**  Simple to use, perfect for anyone starting with OSINT.

## Installation and Setup

### Prerequisites
* Python 3.12 or higher

### Steps
1.  **Install uv:**
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
4.  **Run the Harvester**
    ```bash
    uv run theHarvester
    ```

## Development

### Install Development Dependencies
```bash
uv sync --extra dev
```

### Run Tests
```bash
uv run pytest
```

### Run Linting and Formatting
```bash
uv run ruff check
```
```bash
uv run ruff format
```

## Passive Modules

TheHarvester integrates with numerous passive modules to gather information from various sources:

*   baidu: Baidu search engine
*   bevigil: CloudSEK BeVigil scans mobile application for OSINT assets
*   brave: Brave search engine - now uses official Brave Search API
*   bufferoverun: Fast domain name lookups for TLS certificates in IPv4 space
*   builtwith: Find out what websites are built with
*   censys: Uses certificates searches to enumerate subdomains and gather emails
*   certspotter: Cert Spotter monitors Certificate Transparency logs
*   criminalip: Specialized Cyber Threat Intelligence (CTI) search engine
*   crtsh: Comodo Certificate search
*   dehashed: Take your data security to the next level is
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
*   pentesttools: Cloud-based toolkit for offensive security testing, focused on web applications and network penetration testing
*   projecdiscovery: Actively collects and maintains internet-wide assets data, to enhance research and analyse changes around DNS for better insights
*   rapiddns: DNS query tool which make querying subdomains or sites of a same IP easy
*   rocketreach: Access real-time verified personal/professional emails, phone numbers, and social media links
*   securityscorecard: helps TPRM and SOC teams detect, prioritize, and remediate vendor risk across their entire supplier ecosystem at scale
*   securityTrails: Security Trails search engine, the world's largest repository of historical DNS data
*   -s, --shodan: Shodan search engine will search for ports and banners from discovered hosts
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

*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

For advanced features and broader data collection, some modules require API keys.  Refer to the installation documentation for detailed setup instructions: [API Key Setup](https://github.com/laramies/theHarvester/wiki/Installation#api-keys).

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

## Community and Support

*   **Twitter:**  Follow [@laramies](https://twitter.com/laramies) (Christian Martorella) and cmartorella@edge-security.com, [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) (Matthew Brown), and [@jay_townsend1](https://twitter.com/jay_townsend1) (Jay "L1ghtn1ng" Townsend).

## Contributors

*   [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) - Matthew Brown
*   [@jay_townsend1](https://twitter.com/jay_townsend1) - Jay "L1ghtn1ng" Townsend
*   [@discoverscripts](https://twitter.com/discoverscripts) - Lee Baird

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)
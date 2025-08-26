<!-- theHarvester Logo -->
<img src="https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp" alt="theHarvester Logo" width="200">

[![theHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]()
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]()
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

# theHarvester: Your Go-To OSINT Tool for Reconnaissance

**theHarvester is a powerful open-source intelligence (OSINT) tool designed to automate the early stages of penetration testing and red team assessments, allowing you to discover a target's attack surface quickly.**

[View the Original Repository](https://github.com/laramies/theHarvester)

## Key Features

*   **Comprehensive Information Gathering**: Collects a wealth of information, including:
    *   Subdomains
    *   Email addresses
    *   IP addresses
    *   URLs
    *   Employee names
*   **Multiple Data Sources**: Leverages numerous public resources:
    *   Search engines (Google, Bing, DuckDuckGo, etc.)
    *   Social media platforms (e.g., LinkedIn)
    *   Public databases (e.g., Shodan, Censys)
*   **Active and Passive Modules**:  Supports both passive and active reconnaissance techniques.
*   **API Integration**: Integrates with various APIs for enhanced data retrieval (API keys required for some modules).
*   **Easy to Use**: Simple command-line interface for quick and effective information gathering.

## Installation

**Prerequisites:**
* Python 3.12 or higher.

**Installation Steps:**

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

4.  **Run the Harvester:**
    ```bash
    uv run theHarvester
    ```

## Development

**Install development dependencies:**
```bash
uv sync --extra dev
```

**Run tests:**
```bash
uv run pytest
```

**Run linting and formatting:**
```bash
uv run ruff check
```
```bash
uv run ruff format
```

## Modules

### Passive Modules

TheHarvester integrates with various passive modules to gather information from different sources. Here's a list of supported passive modules:

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

### Active Modules

*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

Some modules require API keys. Detailed API key setup instructions can be found in the [Installation wiki](https://github.com/laramies/theHarvester/wiki/Installation#api-keys)

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

## Get in Touch

*   [Twitter Follow](https://twitter.com/laramies) Christian Martorella @laramies
  cmartorella@edge-security.com
*   [Twitter Follow](https://twitter.com/NotoriousRebel1) Matthew Brown @NotoriousRebel1
*   [Twitter Follow](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend @jay_townsend1

## Main Contributors

*   [Twitter Follow](https://twitter.com/NotoriousRebel1) Matthew Brown @NotoriousRebel1
*   [Twitter Follow](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend @jay_townsend1
*   [Twitter Follow](https://twitter.com/discoverscripts) Lee Baird @discoverscripts

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)
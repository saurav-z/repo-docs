<!-- Improved README.md for theHarvester -->
<div align="center">
  <a href="https://github.com/laramies/theHarvester">
    <img src="https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp" alt="theHarvester Logo" width="200">
  </a>
  <br>
  <a href="https://github.com/laramies/theHarvester">
    <h1>theHarvester: OSINT Reconnaissance Tool</h1>
  </a>
  <p><em>Uncover a domain's attack surface by gathering crucial information from open source intelligence (OSINT) sources.</em></p>
  <br>
  <img src="https://img.shields.io/github/last-commit/laramies/theHarvester" alt="Last Commit">
  <img src="https://img.shields.io/github/stars/laramies/theHarvester?style=social" alt="Stars">
  <img src="https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg" alt="CI Status">
  <img src="https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg" alt="Docker CI Status">
  <a href="https://inventory.raw.pm/">
    <img src="https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg" alt="Rawsec Inventory">
  </a>
</div>

## Overview

**theHarvester** is a powerful, yet easy-to-use OSINT (Open Source Intelligence) tool designed for reconnaissance during penetration testing and red team engagements. It gathers valuable information, including names, emails, IPs, subdomains, and URLs, from numerous public sources to help you understand a target's external threat landscape.

## Key Features

*   **Comprehensive Information Gathering:** Collects a wide range of data points essential for reconnaissance.
*   **Multiple Data Sources:** Leverages various public resources for diverse and thorough information gathering.
*   **Modular Design:** Supports a variety of passive and active modules for flexible data collection.
*   **API Key Integration:** Integrates with numerous third-party services, expanding data gathering capabilities.
*   **Easy to Use:** Simple command-line interface for quick deployment and efficient data collection.

## Installation

### Prerequisites

*   Python 3.12 or higher
*   [Installation Guide](https://github.com/laramies/theHarvester/wiki/Installation)

### Installation using `uv`

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```
3.  **Install dependencies and create a virtual environment:**
    ```bash
    uv sync
    ```
4.  **Run theHarvester:**
    ```bash
    uv run theHarvester
    ```

### Development

*   **Install development dependencies:**
    ```bash
    uv sync --extra dev
    ```

*   **Run tests:**
    ```bash
    uv run pytest
    ```

*   **Run linting and formatting:**
    ```bash
    uv run ruff check
    uv run ruff format
    ```

## Modules

### Passive Modules

*   baidu: Baidu search engine (https://www.baidu.com)
*   bevigil: CloudSEK BeVigil scans mobile application for OSINT assets (https://bevigil.com/osint-api)
*   brave: Brave search engine - now uses official Brave Search API (https://api-dashboard.search.brave.com)
*   bufferoverun: Fast domain name lookups for TLS certificates in IPv4 space (https://tls.bufferover.run)
*   builtwith: Find out what websites are built with (https://builtwith.com)
*   censys: Uses certificates searches to enumerate subdomains and gather emails (https://censys.io)
*   certspotter: Cert Spotter monitors Certificate Transparency logs (https://sslmate.com/certspotter)
*   criminalip: Specialized Cyber Threat Intelligence (CTI) search engine (https://www.criminalip.io)
*   crtsh: Comodo Certificate search (https://crt.sh)
*   dehashed: Take your data security to the next level is (https://dehashed.com)
*   dnsdumpster: Domain research tool that can discover hosts related to a domain (https://dnsdumpster.com)
*   duckduckgo: DuckDuckGo search engine (https://duckduckgo.com)
*   fullhunt: Next-generation attack surface security platform (https://fullhunt.io)
*   github-code: GitHub code search engine (https://www.github.com)
*   hackertarget: Online vulnerability scanners and network intelligence to help organizations (https://hackertarget.com)
*   haveibeenpwned: Check if your email address is in a data breach (https://haveibeenpwned.com)
*   hunter: Hunter search engine (https://hunter.io)
*   hunterhow: Internet search engines for security researchers (https://hunter.how)
*   intelx: Intelx search engine (https://intelx.io)
*   leaklookup: Data breach search engine (https://leak-lookup.com)
*   netlas: A Shodan or Censys competitor (https://app.netlas.io)
*   onyphe: Cyber defense search engine (https://www.onyphe.io)
*   otx: AlienVault open threat exchange (https://otx.alienvault.com)
*   pentesttools: Cloud-based toolkit for offensive security testing, focused on web applications and network penetration testing (https://pentest-tools.com)
*   projecdiscovery: Actively collects and maintains internet-wide assets data, to enhance research and analyse changes around DNS for better insights (https://chaos.projectdiscovery.io)
*   rapiddns: DNS query tool which make querying subdomains or sites of a same IP easy (https://rapiddns.io)
*   rocketreach: Access real-time verified personal/professional emails, phone numbers, and social media links (https://rocketreach.co)
*   securityscorecard: helps TPRM and SOC teams detect, prioritize, and remediate vendor risk across their entire supplier ecosystem at scale (https://securityscorecard.com)
*   securityTrails: Security Trails search engine, the world's largest repository of historical DNS data (https://securitytrails.com)
*   -s, --shodan: Shodan search engine will search for ports and banners from discovered hosts (https://shodan.io)
*   subdomaincenter: A subdomain finder tool used to find subdomains of a given domain (https://www.subdomain.center)
*   subdomainfinderc99: A subdomain finder is a tool used to find the subdomains of a given domain (https://subdomainfinder.c99.nl)
*   threatminer: Data mining for threat intelligence (https://www.threatminer.org)
*   tomba: Tomba search engine (https://tomba.io)
*   urlscan: A sandbox for the web that is a URL and website scanner (https://urlscan.io)
*   venacus: Venacus search engine (https://venacus.com)
*   virustotal: Domain search (https://www.virustotal.com)
*   whoisxml: Subdomain search (https://subdomains.whoisxmlapi.com/api/pricing)
*   yahoo: Yahoo search engine (https://www.yahoo.com)
*   zoomeye: China's version of Shodan (https://www.zoomeye.org)

### Active Modules

*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

## Modules Requiring API Keys

Refer to the [API Keys documentation](https://github.com/laramies/theHarvester/wiki/Installation#api-keys) for setup instructions.

*   bevigil - 50 free queries/month, 1k queries/month $50
*   brave - Free plan available, Pro plans for higher limits
*   bufferoverun - 100 free queries/month, 10k/month $25
*   builtwith - 50 free queries ever, $2950/yr
*   censys - 500 credits $100
*   criminalip - 100 free queries/month, 700k/month $59
*   dehashed - 500 credts $15, 5k credits $150
*   dnsdumpster - 50 free querries/day, $49
*   fullhunt - 50 free queries, 200 queries $29/month, 500 queries $59/month
*   github-code
*   haveibeenpwned - 10 email searches/min $4.50, 50 email searches/min $22
*   hunter - 50 credits/month free, 12k credits/yr $34
*   hunterhow - 10k free API results per 30 days, 50k API results per 30 days $10
*   intelx
*   leaklookup - 20 credits $10, 50 credits $20, 140 credits $50, 300 credits $100
*   netlas - 50 free requests/day, 1k requests $49, 10k requests $249
*   onyphe - 10M results/month $587
*   pentesttools - 5 assets netsec $95/month, 5 assets webnetsec $140/month
*   projecdiscovery - requires work email. Free monthly discovery and vulnerability scans on sign-up email domain, enterprise $
*   rocketreach - 100 email lookups/month $48, 250 email lookups/month $108
*   securityscorecard
*   securityTrails - 50 free queries/month, 20k queries/month $500
*   shodan - Freelancer $69 month, Small Business $359 month
*   tomba - 25 searches/month free, 1k searches/month $39, 5k searches/month $89
*   venacus - 1 search/day free, 10 searches/day $12, 30 searches/day $36
*   whoisxml - 2k queries $50, 5k queries $105
*   zoomeye - 5 results/day free, 30/results/day $190/yr

## Get Involved

*   [Report Bugs and Request Features](https://github.com/laramies/theHarvester/issues)
*   Follow on Twitter:
    *   [@laramies](https://twitter.com/laramies) - Christian Martorella
    *   [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) - Matthew Brown
    *   [@jay_townsend1](https://twitter.com/jay_townsend1) - Jay "L1ghtn1ng" Townsend

## Contributors

*   Matthew Brown - [@NotoriousRebel1](https://twitter.com/NotoriousRebel1)
*   Jay "L1ghtn1ng" Townsend - [@jay_townsend1](https://twitter.com/jay_townsend1)
*   Lee Baird - [@discoverscripts](https://twitter.com/discoverscripts)

## Acknowledgments

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries
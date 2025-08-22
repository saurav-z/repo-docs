[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)](https://github.com/laramies/theHarvester)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/python-app.yml)
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/docker-image.yml)
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## theHarvester: Your OSINT Toolkit for Reconnaissance

**theHarvester is a powerful open-source intelligence (OSINT) tool designed to help penetration testers and red teamers gather crucial information about a target domain.**  This tool helps you uncover a domain's external threat landscape by collecting valuable data from various public sources.

### Key Features:

*   **Comprehensive Information Gathering:** Collects names, emails, IP addresses, subdomains, and URLs.
*   **Multiple Data Sources:** Leverages a wide array of public resources for information retrieval.
*   **Passive and Active Modules:** Offers both passive data collection (without direct interaction) and active modules for deeper analysis.
*   **Easy to Use:** Simple to install and operate, making it accessible for both beginners and experienced users.
*   **Python-Based:** Built on Python, ensuring cross-platform compatibility.
*   **Modular Design:**  Easily extensible to support new data sources.

### Installation and Usage

1.  **Prerequisites:**
    *   Python 3.12 or higher.

2.  **Install `uv` (optional, for faster dependency management):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

4.  **Install Dependencies and Set Up a Virtual Environment:**
    ```bash
    uv sync
    ```

5.  **Run the Tool:**
    ```bash
    uv run theHarvester
    ```

### Development

*   **Install Development Dependencies:**
    ```bash
    uv sync --extra dev
    ```

*   **Run Tests:**
    ```bash
    uv run pytest
    ```

*   **Run Linting and Formatting:**
    ```bash
    uv run ruff check
    uv run ruff format
    ```

### Modules

theHarvester uses a modular design to pull data from a variety of sources.

#### Passive Modules

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

#### Active Modules
*   DNS brute force: dictionary brute force enumeration
*   Screenshots: Take screenshots of subdomains that were found

### Modules Requiring API Keys

Some modules require API keys for access.

Documentation for setting up API keys can be found at: [theHarvester Wiki - Installation](https://github.com/laramies/theHarvester/wiki/Installation#api-keys).

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

### Contact & Contributing

*   **Follow on Twitter:**  [@laramies](https://twitter.com/laramies) (Christian Martorella)
*   **Contact:**  cmartorella@edge-security.com
*   **Follow on Twitter:**  [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) (Matthew Brown)
*   **Follow on Twitter:**  [@jay_townsend1](https://twitter.com/jay_townsend1) (Jay "L1ghtn1ng" Townsend)
*   **Main Contributors:**
    *   [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) Matthew Brown
    *   [@jay_townsend1](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend
    *   [@discoverscripts](https://twitter.com/discoverscripts) Lee Baird

### Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)

**[Visit the theHarvester GitHub Repository](https://github.com/laramies/theHarvester) to get started with OSINT gathering today!**
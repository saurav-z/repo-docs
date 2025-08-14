# Destroylist: Your Defense Against Phishing and Scam Domains

**Protect yourself from online threats with Destroylist, a continuously updated blacklist of malicious domains.** [Explore the original repository](https://github.com/phishdestroy/destroylist).

## Key Features

*   **Real-time Threat Intelligence:** Stay ahead of phishing and scam attempts with a constantly updated list.
*   **Multiple Data Feeds:** Access various data feeds, including primary curated lists, DNS-verified active domains, and community-contributed blocklists.
*   **Community Driven:** Benefit from the collective knowledge of the community to identify and block threats.
*   **Easy Integration:** Data feeds are available in JSON format for seamless integration with firewalls, DNS resolvers, browser extensions, and threat platforms.
*   **Historical Archive:** Access a vast archive of over 500,000+ historical domains for research and analysis.
*   **Appeals Process:** Quickly address any false positives with a dedicated appeals process.

## Data Feeds - Stay Informed

| Data Feed                     | Description                                                              | Link                                                                           |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Primary Curated List          | Core phishing/scam domains, real-time updates                          | [list.json](https://github.com/phishdestroy/destroylist/raw/main/list.json)     |
| Active DNS-Verified           | DNS-checked live threats                                                 | [active\_domains.json](https://github.com/phishdestroy/destroylist/raw/main/dns/active_domains.json) |
| Community General             | Broad aggregated blocklist, hourly updates                               | [blocklist.json](https://github.com/phishdestroy/destroylist/raw/main/community/blocklist.json) |
| Community Live                | DNS-checked active community threats                                     | [live\_blocklist.json](https://github.com/phishdestroy/destroylist/raw/main/community/live_blocklist.json) |

## Update Process - How It Works

1.  **Gather:** Collect phishing domains continuously.
2.  **Sync:** Cross-reference data with trusted sources.
3.  **Add:** Integrate new malicious domains in real-time.
4.  **Clean:** Remove inactive or expired domains.

## Goals & Usage - Who Can Benefit

*   **Security Professionals:** Enhance your security gear with reliable threat data.
*   **Developers:** Integrate Destroylist into your scripts and automation processes.
*   **Researchers:** Conduct in-depth research and analysis of phishing trends.
*   **Security Operations Centers (SOCs):** Improve threat monitoring and incident response.

## Appeals Process - Addressing False Positives

If you believe a domain has been incorrectly listed, please:

*   Submit an [Appeals Form](https://phishdestroy.io/appeals/)
*   Open a GitHub issue with supporting evidence.

## License

This project is licensed under the MIT License, granting you the freedom to use, modify, and distribute the code.

## Join the Fight Against Phishing!

We welcome your contributions to improve Destroylist.  Please submit issues or pull requests with:

*   Detection tweaks
*   Integration tips
*   New threat intelligence
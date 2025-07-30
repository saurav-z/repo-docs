[![Maltrail](https://i.imgur.com/3xjInOD.png)](https://github.com/stamparm/maltrail)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Your Frontline Defense Against Malicious Network Traffic

Maltrail is a powerful, open-source malicious traffic detection system, offering real-time monitoring and alerting against a wide range of threats. Designed to be easily deployed and integrated, it helps you identify and respond to suspicious activity on your network.  [See the original repository here.](https://github.com/stamparm/maltrail)

**Key Features:**

*   **Real-time Threat Detection:** Monitors network traffic for malicious indicators, including domains, URLs, IPs, and HTTP User-Agent headers.
*   **Comprehensive Threat Intelligence:** Utilizes a vast collection of blacklists and feeds from various sources, as well as static trails derived from AV reports and custom lists.
*   **Heuristic Analysis:** Employs advanced heuristics to identify unknown and emerging threats, enhancing detection capabilities.
*   **Flexible Architecture:**  Employs a modular architecture with a sensor, server, and client components.
*   **User-Friendly Reporting Interface:** Provides a web-based interface for easy analysis of detected threats.
*   **Integration with Other Systems:**  Supports integration with popular tools like Syslog and Logstash for centralized logging and analysis.

**Key Features Explained:**

*   **Comprehensive Threat Intelligence:** Maltrail leverages multiple public and private sources to stay on top of new and known threats.
    *   Utilizes publicly available (black)lists:
        *   360bigviktor, 360chinad, 360conficker, 360cryptolocker, 360gameover, 360locky, 360necurs, 360suppobox, 360tofsee, 360virut, abuseipdb, alienvault, atmos, badips, bitcoinnodes, blackbook, blocklist, botscout, bruteforceblocker, ciarmy, cobaltstrike, cruzit, cybercrimetracker, dataplane, dshieldip, emergingthreatsbot, emergingthreatscip, emergingthreatsdns, feodotrackerip, gpfcomics, greensnow, ipnoise, kriskinteldns, kriskintelip, malc0de, malwaredomainlistdns, malwaredomains, maxmind, minerchk, myip, openphish, palevotracker, policeman, pony, proxylists, proxyrss, proxyspy, ransomwaretrackerdns, ransomwaretrackerip, ransomwaretrackerurl, riproxies, rutgers, sblam, socksproxy, sslbl, sslproxies, talosintelligence, torproject, trickbot, turris, urlhaus, viriback, vxvault, zeustrackermonitor, zeustrackerurl, etc.
    *   Static entries included, manually (from AV reports and personal research):
        *   1ms0rry, 404, 9002, aboc, absent, ab, acbackdoor, acridrain, activeagent, adrozek, advisorbot, adwind, adylkuzz, adzok, afrodita, agaadex, agenttesla, aldibot, alina, allakore, almalocker, almashreq, alpha, alureon, amadey, amavaldo, amend\_miner, ammyyrat, android\_acecard, android\_actionspy, android\_adrd, android\_ahmythrat, android\_alienspy, android\_andichap, android\_androrat, android\_anubis, android\_arspam, android\_asacub, android\_backflash, android\_bankbot, android\_bankun, android\_basbanke, android\_basebridge, android\_besyria, android\_blackrock, android\_boxer, android\_buhsam, android\_busygasper, android\_calibar, android\_callerspy, android\_camscanner, android\_cerberus, android\_chuli, android\_circle, android\_claco, android\_clickfraud, android\_cometbot, android\_cookiethief, android\_coolreaper, android\_copycat, android\_counterclank, android\_cyberwurx, android\_darkshades, android\_dendoroid, android\_dougalek, android\_droidjack, android\_droidkungfu, android\_enesoluty, android\_eventbot, android\_ewalls, android\_ewind, android\_exodus, android\_exprespam, android\_fakeapp, android\_fakebanco, android\_fakedown, android\_fakeinst, android\_fakelog, android\_fakemart, android\_fakemrat, android\_fakeneflic, android\_fakesecsuit, android\_fanta, android\_feabme, android\_flexispy, android\_fobus, android\_fraudbot, android\_friend, android\_frogonal, android\_funkybot, android\_gabas, android\_geinimi, android\_generic, android\_geost, android\_ghostpush, android\_ginmaster, android\_ginp, android\_gmaster, android\_gnews, android\_godwon, android\_golddream, android\_goldencup, android\_golfspy, android\_gonesixty, android\_goontact, android\_gplayed, android\_gustuff, android\_gypte, android\_henbox, android\_hiddad, android\_hydra, android\_ibanking, android\_joker, android\_jsmshider, android\_kbuster, android\_kemoge, android\_ligarat, android\_lockdroid, android\_lotoor, android\_lovetrap, android\_malbus, android\_mandrake, android\_maxit, android\_mobok, android\_mobstspy, android\_monokle, android\_notcompatible, android\_oneclickfraud, android\_opfake, android\_ozotshielder, android\_parcel, android\_phonespy, android\_pikspam, android\_pjapps, android\_qdplugin, android\_raddex, android\_ransomware, android\_redalert, android\_regon, android\_remotecode, android\_repane, android\_riltok, android\_roamingmantis, android\_roidsec, android\_rotexy, android\_samsapo, android\_sandrorat, android\_selfmite, android\_shadowvoice, android\_shopper, android\_simbad, android\_simplocker, android\_skullkey, android\_sndapps, android\_spynote, android\_spytekcell, android\_stels, android\_svpeng, android\_swanalitics, android\_teelog, android\_telerat, android\_tetus, android\_thiefbot, android\_tonclank, android\_torec, android\_triada, android\_uracto, android\_usbcleaver, android\_viceleaker, android\_vmvol, android\_walkinwat, android\_windseeker, android\_wirex, android\_wolfrat, android\_xavirad, android\_xbot007, android\_xerxes, android\_xhelper, android\_xploitspy, android\_z3core, android\_zertsecurity, android\_ztorg, andromeda, antefrigus, antibot, anubis, anuna, apocalypse, apt\_12, apt\_17, apt\_18, apt\_23, apt\_27, apt\_30, apt\_33, apt\_37, apt\_38, apt\_aridviper, apt\_babar, apt\_bahamut, etc.

*   **Architecture**
    *   **Traffic** -&gt; **Sensor** &lt;-&gt; **Server** &lt;-&gt; **Client** architecture
    *   **Sensor:** Runs on the monitoring node, passively or inline. Monitors traffic for blacklisted items. Sends events to the server.
    *   **Server:** Stores event details and provides a reporting interface.  Can be skipped, with events stored locally by the sensor.
    *   **Client:** Web-based reporting application for data presentation.

### Sections:

*   [Introduction](#introduction)
*   [Architecture](#architecture)
*   [Demo pages](#demo-pages)
*   [Requirements](#requirements)
*   [Quick start](#quick-start)
*   [Administrator's guide](#administrators-guide)
     *   [Sensor](#sensor)
     *   [Server](#server)
*   [User's guide](#users-guide)
     *   [Reporting interface](#reporting-interface)
*   [Real-life cases](#real-life-cases)
     *   [Mass scans](#mass-scans)
     *   [Anonymous attackers](#anonymous-attackers)
     *   [Service attackers](#service-attackers)
     *   [Malware](#malware)
     *   [Suspicious domain lookups](#suspicious-domain-lookups)
     *   [Suspicious ipinfo requests](#suspicious-ipinfo-requests)
     *   [Suspicious direct file downloads](#suspicious-direct-file-downloads)
     *   [Suspicious HTTP requests](#suspicious-http-requests)
     *   [Port scanning](#port-scanning)
     *   [DNS resource exhaustion](#dns-resource-exhaustion)
     *   [Data leakage](#data-leakage)
     *   [False positives](#false-positives)
*   [Best practice(s)](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank you](#thank-you)
*   [Third-party integrations](#third-party-integrations)

### Demo Pages
Fully functional demo pages with collected real-life threats can be found [here](https://maltraildemo.github.io/).

### Requirements

*   Python 2.6, 2.7, or 3.x
*   \*nix/BSD system
*   pcapy-ng package
*   Sensor: at least 1GB of RAM
*   Sensor: Administrative/root privileges

### Quick Start

*   Instructions for Ubuntu/Debian, SUSE/openSUSE, and Docker installations are provided in the original README.  See above.

### Administrator's Guide

*   Detailed configuration options for the sensor and server components, including logging, security, and integration.

### User's Guide

*   Explains the reporting interface's features, including timeline, summary, and threat details.

### Real-Life Cases

*   Illustrates Maltrail's capabilities with examples of real-world threats, such as mass scans, malware, and suspicious activity.

### Best Practices

*   Provides recommended installation and configuration steps.

### License

*   MIT License. See [LICENSE](https://github.com/stamparm/maltrail/blob/master/LICENSE) for more details.

### Sponsors

*   [Sansec](https://sansec.io/) (2024-)
*   [Sansec](https://sansec.io/) (2020-2021)

### Developers

*   Miroslav Stampar ([@stamparm](https://github.com/stamparm))
*   Mikhail Kasimov ([@MikhailKasimov](https://github.com/MikhailKasimov))

### Presentations

*   47th TF-CSIRT Meeting, Prague (Czech Republic), 2016 ([slides](https://www.terena.org/activities/tf-csirt/meeting47/M.Stampar-Maltrail.pdf))

### Publications

*   Detect attacks on your network with Maltrail, Linux Magazine, 2022 ([Annotation](https://www.linux-magazine.com/Issues/2022/258/Maltrail))
*   Best Cyber Threat Intelligence Feeds ([SilentPush Review, 2022](https://www.silentpush.com/blog/best-cyber-threat-intelligence-feeds))
*   Research on Network Malicious Traffic Detection System Based on Maltrail ([Nanotechnology Perceptions, ISSN 1660-6795, 2024](https://nano-ntp.com/index.php/nano/article/view/1915/1497))

### Blacklist

*   Maltrail's daily updated blacklist of malware-related domains can be found [here](https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt). It is based on trails found at [trails/static/malware](trails/static/malware) and can be safely used for DNS traffic blocking purposes.

### Thank You

*   (List of contributors)

### Third-Party Integrations

*   (List of integrations)
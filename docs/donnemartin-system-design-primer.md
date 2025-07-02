# System Design Primer: Your Guide to Designing Scalable Systems

**Learn how to design large-scale systems and ace your system design interview with this comprehensive and community-driven resource.**

[View the original repository on GitHub](https://github.com/donnemartin/system-design-primer).

## Key Features

*   **Organized Resources:** An extensive collection of resources, articles, and examples to help you understand and build scalable systems.
*   **Interview Prep:** Prepare for system design interviews with a structured study guide, common questions, and sample solutions (discussions, code, and diagrams).
*   **Community-Driven:** Contribute to this open-source project and learn from the community.
*   **Anki Flashcards:** Memorize key concepts using the provided Anki flashcard decks (System Design, System Design Exercises, and Object Oriented Design Exercises).
*   **Interactive Coding Challenges:** Resources to help you prep for the coding interview with the sister repo [Interactive Coding Challenges](https://github.com/donnemartin/interactive-coding-challenges), which also contains an Anki deck.

## Table of Contents

*   [Motivation](#motivation)
*   [Anki Flashcards](#anki-flashcards)
*   [Contributing](#contributing)
*   [Index of System Design Topics](#index-of-system-design-topics)
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
*   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [System Design Topics: Start Here](#system-design-topics-start-here)
*   [Performance vs Scalability](#performance-vs-scalability)
*   [Latency vs Throughput](#latency-vs-throughput)
*   [Availability vs Consistency](#availability-vs-consistency)
    *   [CAP Theorem](#cap-theorem)
        *   [CP - Consistency and Partition Tolerance](#cp---consistency-and-partition-tolerance)
        *   [AP - Availability and Partition Tolerance](#ap---availability-and-partition-tolerance)
*   [Consistency Patterns](#consistency-patterns)
    *   [Weak Consistency](#weak-consistency)
    *   [Eventual Consistency](#eventual-consistency)
    *   [Strong Consistency](#strong-consistency)
*   [Availability Patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
        *   [Active-passive](#active-passive)
        *   [Active-active](#active-active)
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
*   [Domain Name System](#domain-name-system)
*   [Content Delivery Network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load Balancer](#load-balancer)
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
    *   [Layer 4 load balancing](#layer-4-load-balancing)
    *   [Layer 7 load balancing](#layer-7-load-balancing)
    *   [Horizontal scaling](#horizontal-scaling)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Load Balancer vs Reverse Proxy](#load-balancer-vs-reverse-proxy)
*   [Application Layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service Discovery](#service-discovery)
*   [Database](#database)
    *   [Relational Database Management System (RDBMS)](#relational-database-management-system-rdbms)
        *   [Master-slave replication](#master-slave-replication)
        *   [Master-master replication](#master-master-replication)
        *   [Federation](#federation)
        *   [Sharding](#sharding)
        *   [Denormalization](#denormalization)
        *   [SQL Tuning](#sql-tuning)
    *   [NoSQL](#nosql)
        *   [Key-value store](#key-value-store)
        *   [Document store](#document-store)
        *   [Wide column store](#wide-column-store)
        *   [Graph Database](#graph-database)
    *   [SQL or NoSQL](#sql-or-nosql)
*   [Cache](#cache)
    *   [Client caching](#client-caching)
    *   [CDN caching](#cdn-caching)
    *   [Web server caching](#web-server-caching)
    *   [Database caching](#database-caching)
    *   [Application caching](#application-caching)
    *   [Caching at the database query level](#caching-at-the-database-query-level)
    *   [Caching at the object level](#caching-at-the-object-level)
    *   [When to update the cache](#when-to-update-the-cache)
        *   [Cache-aside](#cache-aside)
        *   [Write-through](#write-through)
        *   [Write-behind (write-back)](#write-behind-write-back)
        *   [Refresh-ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message queues](#message-queues)
    *   [Task queues](#task-queues)
    *   [Back pressure](#back-pressure)
*   [Communication](#communication)
    *   [Transmission Control Protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User Datagram Protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote Procedure Call (RPC)](#remote-procedure-call-rpc)
    *   [Representational State Transfer (REST)](#representational-state-transfer-rest)
*   [Security](#security)
*   [Appendix](#appendix)
    *   [Powers of Two Table](#powers-of-two-table)
    *   [Latency Numbers Every Programmer Should Know](#latency-numbers-every-programmer-should-know)
    *   [Additional System Design Interview Questions](#additional-system-design-interview-questions)
    *   [Real World Architectures](#real-world-architectures)
    *   [Company Architectures](#company-architectures)
    *   [Company Engineering Blogs](#company-engineering-blogs)
*   [Under Development](#under-development)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

## Motivation

*   Learn how to design large-scale systems.
*   Prep for the system design interview.

## How to Approach a System Design Interview Question

*   **Step 1:** Outline use cases, constraints, and assumptions.
*   **Step 2:** Create a high-level design.
*   **Step 3:** Design core components.
*   **Step 4:** Scale the design.

## System Design Interview Questions with Solutions

Solutions linked to content in the `solutions/` folder.

| Question |  |
|---|---|
| Design Pastebin.com (or Bit.ly) | [Solution](solutions/system_design/pastebin/README.md) |
| Design the Twitter timeline and search (or Facebook feed and search) | [Solution](solutions/system_design/twitter/README.md) |
| Design a web crawler | [Solution](solutions/system_design/web_crawler/README.md) |
| Design Mint.com | [Solution](solutions/system_design/mint/README.md) |
| Design the data structures for a social network | [Solution](solutions/system_design/social_graph/README.md) |
| Design a key-value store for a search engine | [Solution](solutions/system_design/query_cache/README.md) |
| Design Amazon's sales ranking by category feature | [Solution](solutions/system_design/sales_rank/README.md) |
| Design a system that scales to millions of users on AWS | [Solution](solutions/system_design/scaling_aws/README.md) |
| Add a system design question | [Contribute](#contributing) |
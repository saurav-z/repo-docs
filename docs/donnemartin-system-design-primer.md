# System Design Primer: Your Guide to Building Scalable Systems

**Learn how to design large-scale systems and ace your system design interviews with this comprehensive, open-source resource. Get started with the [original repo](https://github.com/donnemartin/system-design-primer)!**

## Key Features

*   **Comprehensive Coverage:** Explore core system design concepts, including scalability, performance, consistency, availability, and more.
*   **Interview Preparation:** Master system design interview questions with sample discussions, code, and diagrams.
*   **Open-Source Community:** Learn from the community and contribute to the project.
*   **Practical Resources:** Benefit from Anki flashcards, real-world architectures, and company engineering blogs.

## Core Concepts

*   **Performance vs. Scalability:** Understanding the trade-offs between optimizing for speed and handling increased load.
*   **Latency vs. Throughput:** Balancing the time to complete a task with the number of tasks completed over time.
*   **CAP Theorem:** Grasping the fundamental trade-offs between consistency, availability, and partition tolerance in distributed systems.
*   **Consistency Patterns:** Implementing approaches such as eventual consistency and strong consistency.
*   **Availability Patterns:** Utilizing techniques such as failover and replication to ensure system uptime.
*   **Key Components:** Delving into the intricacies of DNS, CDNs, load balancers, application layers, databases, and caching strategies.
*   **Communication Protocols:** Exploring the ins and outs of HTTP, TCP, UDP, REST, and RPC.
*   **Practical Guides:** Covering system design topics to help you ace your interviews and build scalable systems.

## Table of Contents

*   [Key Features](#key-features)
*   [Core Concepts](#core-concepts)
*   [Table of Contents](#table-of-contents)
*   [System Design Topics: Start Here](#system-design-topics-start-here)
*   [Performance vs Scalability](#performance-vs-scalability)
*   [Latency vs Throughput](#latency-vs-throughput)
*   [Availability vs Consistency](#availability-vs-consistency)
    *   [CAP Theorem](#cap-theorem)
    *   [CP - consistency and partition tolerance](#cp---consistency-and-partition-tolerance)
    *   [AP - availability and partition tolerance](#ap---availability-and-partition-tolerance)
*   [Consistency Patterns](#consistency-patterns)
    *   [Weak consistency](#weak-consistency)
    *   [Eventual consistency](#eventual-consistency)
    *   [Strong consistency](#strong-consistency)
*   [Availability Patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
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
*   [Reverse Proxy (web server)](#reverse-proxy-web-server)
    *   [Load balancer vs reverse proxy](#load-balancer-vs-reverse-proxy)
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
        *   [SQL tuning](#sql-tuning)
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
    *   [Transmission control protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User datagram protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote procedure call (RPC)](#remote-procedure-call-rpc)
    *   [Representational state transfer (REST)](#representational-state-transfer-rest)
*   [Security](#security)
*   [Appendix](#appendix)
    *   [Powers of two table](#powers-of-two-table)
    *   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
    *   [Additional system design interview questions](#additional-system-design-interview-questions)
    *   [Real world architectures](#real-world-architectures)
    *   [Company architectures](#company-architectures)
    *   [Company engineering blogs](#company-engineering-blogs)
*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions with solutions](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)
*   [Under development](#under-development)
*   [Credits](#credits)
*   [Contact info](#contact-info)
*   [License](#license)

## System Design Topics: Start Here

*   **Scalability Video Lecture:** Review the foundational principles of scalable systems ([Harvard Lecture](https://www.youtube.com/watch?v=-W9F__D3oY4)).
*   **Scalability Article:** Deep dive into scalability fundamentals ([lecloud.net](https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono)).

## Study Guide

*   **Short Timeline:** Breadth, some interview questions
*   **Medium Timeline:** Breadth & depth, many interview questions
*   **Long Timeline:** Breadth & more depth, most interview questions

| Time | Topics to Review |
|---|---|
| All | [System design topics](#index-of-system-design-topics) |
| All | [Company engineering blogs](#company-engineering-blogs) |
| All | [Real world architectures](#real-world-architectures) |
| All | [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) |
| Some | [System design interview questions with solutions](#system-design-interview-questions-with-solutions) |
| Some | [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) |
| Some | [Additional system design interview questions](#additional-system-design-interview-questions) |

## How to Approach a System Design Interview Question

1.  **Use Cases, Constraints, and Assumptions:** Define the scope.
2.  **High-Level Design:** Sketch the main components.
3.  **Core Component Design:** Dive into component details.
4.  **Scaling the Design:** Address bottlenecks.

### Back-of-the-Envelope Calculations

*   Refer to the [Appendix](#appendix) for calculations.

## System Design Interview Questions with Solutions

Common questions with discussions, code, and diagrams.

*   [Design Pastebin.com (or Bit.ly)](#design-pastebincom-or-bitly)
*   [Design the Twitter timeline and search (or Facebook feed and search)](#design-the-twitter-timeline-and-search-or-facebook-feed-and-search)
*   [Design a web crawler](#design-a-web-crawler)
*   [Design Mint.com](#design-mintcom)
*   [Design the data structures for a social network](#design-the-data-structures-for-a-social-network)
*   [Design a key-value store for a search engine](#design-a-key-value-store-for-a-search-engine)
*   [Design Amazon's sales ranking by category feature](#design-amazons-sales-ranking-by-category-feature)
*   [Design a system that scales to millions of users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)

## Object-oriented Design Interview Questions with Solutions

*   **Design a hash map**
*   **Design a least recently used cache**
*   **Design a call center**
*   **Design a deck of cards**
*   **Design a parking lot**
*   **Design a chat server**

## Additional Resources

*   [Additional system design interview questions](#additional-system-design-interview-questions)
*   [Real-world architectures](#real-world-architectures)
*   [Company architectures](#company-architectures)
*   [Company engineering blogs](#company-engineering-blogs)
*   [Anki flashcards](https://apps.ankiweb.net/)

---

**This README is a living document. Your contributions are welcome!**
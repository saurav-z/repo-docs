# System Design Primer: Your Guide to Designing Large-Scale Systems

**Master system design concepts and ace your technical interviews with this comprehensive, open-source resource. Explore design patterns, architectural choices, and real-world examples. [Explore the original repo](https://github.com/donnemartin/system-design-primer).**

---

## Key Features:

*   **Comprehensive Coverage:** Dive into essential system design topics.
*   **Interview Prep:** Prepare for system design interviews with practice questions and solutions.
*   **Community Driven:** Benefit from a continuously updated, open-source project with contributions welcome!
*   **Anki Flashcards:** Reinforce learning with provided Anki flashcard decks.
*   **Interactive Coding Challenges:** Complement your learning with the sister repo [Interactive Coding Challenges](https://github.com/donnemartin/interactive-coding-challenges).

---

## Table of Contents

*   [Motivation](#motivation)
    *   [Learn How to Design Large-Scale Systems](#learn-how-to-design-large-scale-systems)
    *   [Learn from the Open Source Community](#learn-from-the-open-source-community)
    *   [Prep for the System Design Interview](#prep-for-the-system-design-interview)
*   [Anki Flashcards](#anki-flashcards)
    *   [Coding Resource: Interactive Coding Challenges](#coding-resource-interactive-coding-challenges)
*   [Contributing](#contributing)
*   [Index of System Design Topics](#index-of-system-design-topics)
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
    *   [Design Pastebin.com (or Bit.ly)](#design-pastebincom-or-bitly)
    *   [Design the Twitter Timeline and Search (or Facebook Feed and Search)](#design-the-twitter-timeline-and-search-or-facebook-feed-and-search)
    *   [Design a Web Crawler](#design-a-web-crawler)
    *   [Design Mint.com](#design-mintcom)
    *   [Design the Data Structures for a Social Network](#design-the-data-structures-for-a-social-network)
    *   [Design a Key-Value Store for a Search Engine](#design-a-key-value-store-for-a-search-engine)
    *   [Design Amazon's Sales Ranking by Category Feature](#design-amazons-sales-ranking-by-category-feature)
    *   [Design a System that Scales to Millions of Users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)
*   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [System Design Topics: Start Here](#system-design-topics-start-here)
    *   [Step 1: Review the Scalability Video Lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the Scalability Article](#step-2-review-the-scalability-article)
    *   [Next Steps](#next-steps)
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
    *   [Availability in Numbers](#availability-in-numbers)
*   [Domain Name System](#domain-name-system)
*   [Content Delivery Network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load Balancer](#load-balancer)
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
    *   [Layer 4 Load Balancing](#layer-4-load-balancing)
    *   [Layer 7 Load Balancing](#layer-7-load-balancing)
    *   [Horizontal Scaling](#horizontal-scaling)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Load Balancer vs Reverse Proxy](#load-balancer-vs-reverse-proxy)
*   [Application Layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service Discovery](#service-discovery)
*   [Database](#database)
    *   [Relational Database Management System (RDBMS)](#relational-database-management-system-rdbms)
        *   [Master-slave Replication](#master-slave-replication)
        *   [Master-master Replication](#master-master-replication)
        *   [Federation](#federation)
        *   [Sharding](#sharding)
        *   [Denormalization](#denormalization)
        *   [SQL Tuning](#sql-tuning)
    *   [NoSQL](#nosql)
        *   [Key-value Store](#key-value-store)
        *   [Document Store](#document-store)
        *   [Wide Column Store](#wide-column-store)
        *   [Graph Database](#graph-database)
    *   [SQL or NoSQL](#sql-or-nosql)
*   [Cache](#cache)
    *   [Client Caching](#client-caching)
    *   [CDN Caching](#cdn-caching)
    *   [Web Server Caching](#web-server-caching)
    *   [Database Caching](#database-caching)
    *   [Application Caching](#application-caching)
    *   [Caching at the Database Query Level](#caching-at-the-database-query-level)
    *   [Caching at the Object Level](#caching-at-the-object-level)
    *   [When to Update the Cache](#when-to-update-the-cache)
        *   [Cache-aside](#cache-aside)
        *   [Write-through](#write-through)
        *   [Write-behind (Write-back)](#write-behind-write-back)
        *   [Refresh-ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message Queues](#message-queues)
    *   [Task Queues](#task-queues)
    *   [Back Pressure](#back-pressure)
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

---

## Motivation

This repository is your go-to resource for learning and mastering system design, covering everything from fundamental principles to real-world architectures.

### Learn How to Design Large-Scale Systems

Understanding how to design scalable systems will help you become a better engineer.

System design is a broad topic and a *vast amount of resources* are scattered across the web on system design principles.

This repo is an *organized collection* of resources to help you learn how to build systems at scale.

### Learn from the Open Source Community

This is a continually updated, open-source project.

[Contributions](#contributing) are welcome!

### Prep for the System Design Interview

In addition to coding interviews, system design is a **required component** of the **technical interview process** at many tech companies.

**Practice common system design interview questions** and **compare** your results with **sample solutions**: discussions, code, and diagrams.

Additional topics for interview prep:

*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions, **with solutions**](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions, **with solutions**](#object-oriented-design-interview-questions-with-solutions)
*   [Additional system design interview questions](#additional-system-design-interview-questions)
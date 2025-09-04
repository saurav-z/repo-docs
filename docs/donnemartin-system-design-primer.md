# System Design Primer: Your Guide to Designing Large-Scale Systems

**Learn how to design large-scale systems, prepare for system design interviews, and master the core concepts with this comprehensive, community-driven resource. [Explore the original repo](https://github.com/donnemartin/system-design-primer).**

This repository provides a wealth of information to help you:

*   Understand the fundamentals of system design.
*   Prepare for system design interview questions.
*   Learn from a growing open-source community.

## Key Features

*   **Comprehensive Topics:** Explore key concepts with summaries, pros, and cons of various system design topics.
*   **Interview Prep:** Includes a study guide, an approach to system design interview questions, and common interview questions with solutions.
*   **Community Driven:**  A continuously updated open-source project, with contributions from engineers worldwide.
*   **Anki Flashcards:** Use Anki flashcards to retain system design concepts through spaced repetition.
*   **Interactive Coding Challenges:** Explore the sister repo [**Interactive Coding Challenges**](https://github.com/donnemartin/interactive-coding-challenges) for coding interview practice.

## Table of Contents

*   [Index of System Design Topics](#index-of-system-design-topics)
    *   [System Design Topics: Start Here](#system-design-topics-start-here)
        *   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
        *   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
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
        *   [Fail-Over](#fail-over)
            *   [Active-Passive](#active-passive)
            *   [Active-Active](#active-active)
        *   [Replication](#replication)
        *   [Availability in Numbers](#availability-in-numbers)
    *   [Domain Name System](#domain-name-system)
    *   [Content Delivery Network](#content-delivery-network)
        *   [Push CDNs](#push-cdns)
        *   [Pull CDNs](#pull-cdns)
    *   [Load Balancer](#load-balancer)
        *   [Layer 4 Load Balancing](#layer-4-load-balancing)
        *   [Layer 7 Load Balancing](#layer-7-load-balancing)
        *   [Horizontal Scaling](#horizontal-scaling)
    *   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Application Layer](#application-layer)
        *   [Microservices](#microservices)
        *   [Service Discovery](#service-discovery)
    *   [Database](#database)
        *   [Relational Database Management System (RDBMS)](#relational-database-management-system-rdbms)
            *   [Master-Slave Replication](#master-slave-replication)
            *   [Master-Master Replication](#master-master-replication)
            *   [Federation](#federation)
            *   [Sharding](#sharding)
            *   [Denormalization](#denormalization)
            *   [SQL Tuning](#sql-tuning)
        *   [NoSQL](#nosql)
            *   [Key-Value Store](#key-value-store)
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
            *   [Cache-Aside](#cache-aside)
            *   [Write-Through](#write-through)
            *   [Write-Behind (Write-Back)](#write-behind-write-back)
            *   [Refresh-Ahead](#refresh-ahead)
    *   [Asynchronism](#asynchronism)
        *   [Message Queues](#message-queues)
        *   [Task Queues](#task-queues)
        *   [Back Pressure](#back-pressure)
    *   [Communication](#communication)
        *   [Hypertext Transfer Protocol (HTTP)](#hypertext-transfer-protocol-http)
        *   [Transmission Control Protocol (TCP)](#transmission-control-protocol-tcp)
        *   [User Datagram Protocol (UDP)](#user-datagram-protocol-udp)
        *   [Remote Procedure Call (RPC)](#remote-procedure-call-rpc)
        *   [Representational State Transfer (REST)](#representational-state-transfer-rest)
    *   [Security](#security)
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
    *   [Step 1: Outline Use Cases, Constraints, and Assumptions](#step-1-outline-use-cases-constraints-and-assumptions)
    *   [Step 2: Create a High Level Design](#step-2-create-a-high-level-design)
    *   [Step 3: Design Core Components](#step-3-design-core-components)
    *   [Step 4: Scale the Design](#step-4-scale-the-design)
    *   [Back-of-the-Envelope Calculations](#back-of-the-envelope-calculations)
    *   [Source(s) and Further Reading](#sources-and-further-reading)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
*   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [Additional System Design Interview Questions](#additional-system-design-interview-questions)
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
```
*   **How to Contribute:**  See the [Contributing section](#contributing)
*   **Translations:** Support for multiple languages is available. [Add a translation](https://github.com/donnemartin/system-design-primer/issues/28)

---
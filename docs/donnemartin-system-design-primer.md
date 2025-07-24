# System Design Primer: Your Guide to Building Scalable Systems

**Learn how to design and build large-scale systems and ace your system design interviews by leveraging this comprehensive, community-driven resource!**

[Original Repo](https://github.com/donnemartin/system-design-primer) | [Translations](https://github.com/donnemartin/system-design-primer/issues/28)

## Key Features

*   **Comprehensive Coverage:** Explore core system design concepts and patterns.
*   **Interview Preparation:** Master the system design interview process with practical questions, solutions, and guidance.
*   **Community-Driven:** Learn from an actively maintained open-source project with contributions welcome.
*   **Anki Flashcards:**  Reinforce your knowledge with spaced repetition flashcards.
*   **Interactive Coding Challenges:**  Sharpen your coding skills with practice challenges (sister repo: [Interactive Coding Challenges](https://github.com/donnemartin/interactive-coding-challenges)).

## What You'll Learn

This primer helps you learn how to design scalable systems and prepare for system design interviews at leading tech companies.

### Core Concepts

*   Performance vs. Scalability
*   Latency vs. Throughput
*   Availability vs. Consistency (CAP Theorem)
*   Consistency Patterns (Weak, Eventual, Strong)
*   Availability Patterns (Fail-over, Replication)

### System Design Topics

*   Domain Name System (DNS)
*   Content Delivery Network (CDN)
*   Load Balancers
*   Reverse Proxies
*   Application Layer (Microservices, Service Discovery)
*   Databases (RDBMS, NoSQL)
*   Caching
*   Asynchronism (Message Queues)
*   Communication (TCP, UDP, RPC, REST)
*   Security

### Interview Preparation

*   **Study Guide:** Tailor your preparation based on your interview timeline (short, medium, long).
*   **How to Approach System Design Questions:** A step-by-step process to guide your interview discussions.
*   **Sample Questions and Solutions:** Comprehensive walkthroughs of common interview questions.
*   **Object-Oriented Design:** Practice your design skills with object-oriented design exercises.
*   **Real-World Architectures:** Gain insights from architectures used by leading tech companies.
*   **Company Engineering Blogs:** Discover valuable resources for company-specific interview preparation.

## Study Guide

This primer includes a targeted study guide to assist in your preparation. Review the different topics based on your interview timeline:

*   **Short timeline:** Focus on breadth and practice a few interview questions.
*   **Medium timeline:** Aim for breadth and some depth with interview question practice.
*   **Long timeline:** Aim for breadth, in-depth knowledge, and solving most interview questions.

## Index of System Design Topics

### [System Design Topics: Start Here](#system-design-topics-start-here)
*   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
*   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
*   [Next steps](#next-steps)
### [Performance vs scalability](#performance-vs-scalability)
### [Latency vs throughput](#latency-vs-throughput)
### [Availability vs consistency](#availability-vs-consistency)
    *   [CAP theorem](#cap-theorem)
        *   [CP - consistency and partition tolerance](#cp---consistency-and-partition-tolerance)
        *   [AP - availability and partition tolerance](#ap---availability-and-partition-tolerance)
### [Consistency patterns](#consistency-patterns)
    *   [Weak consistency](#weak-consistency)
    *   [Eventual consistency](#eventual-consistency)
    *   [Strong consistency](#strong-consistency)
### [Availability patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
### [Domain name system](#domain-name-system)
### [Content delivery network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
### [Load balancer](#load-balancer)
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
    *   [Layer 4 load balancing](#layer-4-load-balancing)
    *   [Layer 7 load balancing](#layer-7-load-balancing)
    *   [Horizontal scaling](#horizontal-scaling)
### [Reverse proxy (web server)](#reverse-proxy-web-server)
    *   [Load balancer vs reverse proxy](#load-balancer-vs-reverse-proxy)
### [Application layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service discovery](#service-discovery)
### [Database](#database)
    *   [Relational database management system (RDBMS)](#relational-database-management-system-rdbms)
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
### [Cache](#cache)
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
### [Asynchronism](#asynchronism)
    *   [Message queues](#message-queues)
    *   [Task queues](#task-queues)
    *   [Back pressure](#back-pressure)
### [Communication](#communication)
    *   [Transmission control protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User datagram protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote procedure call (RPC)](#remote-procedure-call-rpc)
    *   [Representational state transfer (REST)](#representational-state-transfer-rest)
### [Security](#security)
### [Appendix](#appendix)
    *   [Powers of two table](#powers-of-two-table)
    *   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
    *   [Additional system design interview questions](#additional-system-design-interview-questions)
    *   [Real world architectures](#real-world-architectures)
    *   [Company architectures](#company-architectures)
    *   [Company engineering blogs](#company-engineering-blogs)

## How to Approach System Design Questions

This section will provide a great approach to guide your discussion during a system design interview.  This includes outlining use cases, creating a high-level design, scaling the design, and addressing bottlenecks.

## System Design Interview Questions with Solutions

Explore common system design interview questions with detailed discussions, code, and diagrams.

*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

## Object-Oriented Design Interview Questions with Solutions

Learn about Object-Oriented design interview questions to help you prepare for the interview.

## Contributions

Learn from the community, and help by submitting pull requests to fix errors, add sections or translate the guide!
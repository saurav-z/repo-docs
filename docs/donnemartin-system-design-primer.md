# System Design Primer: Your Ultimate Guide to Designing Scalable Systems

**Master system design principles and ace your technical interviews with this comprehensive, community-driven resource.**  [Explore the original repository](https://github.com/donnemartin/system-design-primer) for in-depth knowledge and hands-on practice.

---

## Key Features

*   **Comprehensive Coverage:** Explore a wide range of system design topics, from fundamental concepts to advanced architectures.
*   **Interview Preparation:**  Learn how to approach system design interview questions, with sample solutions, discussions, code, and diagrams.
*   **Community-Driven:**  Benefit from a continually updated, open-source project, with contributions from the open-source community.
*   **Anki Flashcards:**  Reinforce your learning with pre-built Anki flashcard decks for efficient knowledge retention.
*   **Real-World Architectures:**  Dive into real-world system designs from industry leaders.

---

## Introduction

This repository is an organized collection of resources to help you learn how to design systems at scale. System design is a broad topic, and this primer provides an organized approach to learning the key concepts and principles. Whether you're preparing for a technical interview or seeking to enhance your engineering skills, this primer provides a solid foundation.

---

## Table of Contents

*   [Motivation](#motivation)
    *   [Learn How to Design Large-Scale Systems](#learn-how-to-design-large-scale-systems)
    *   [Learn From the Open Source Community](#learn-from-the-open-source-community)
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
    *   [Design a System That Scales to Millions of Users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)
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
        *   [Cache-aside](#cache-aside)
        *   [Write-through](#write-through)
        *   [Write-behind (Write-back)](#write-behind-write-back)
        *   [Refresh-ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message Queues](#message-queues)
    *   [Task Queues](#task-queues)
    *   [Back Pressure](#back-pressure)
*   [Communication](#communication)
    *   [HTTP](#hypertext-transfer-protocol-http)
    *   [TCP](#transmission-control-protocol-tcp)
    *   [UDP](#user-datagram-protocol-udp)
    *   [RPC](#remote-procedure-call-rpc)
    *   [REST](#representational-state-transfer-rest)
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

This repository provides a structured approach to learning system design principles and preparing for system design interviews.

### Learn How to Design Large-Scale Systems

Understanding how to design scalable systems is critical for any engineer, but the resources available can be scattered.  This primer offers an organized collection of resources to help you master the art of building systems that can handle increased loads.

### Learn From the Open Source Community

This is an open-source project.

[Contributions](#contributing) from the community are welcome.

### Prep for the System Design Interview

System design is a key component of the technical interview process at many tech companies. This primer will prepare you for the system design interview.

---

## Index of System Design Topics

The primer provides summaries of various system design topics, including discussions of pros and cons.

---

## Study Guide

A suggested path through the topics is provided depending on your interview timeline, experience, what positions you are interviewing for, and which companies you are interviewing with.

---

## How to Approach a System Design Interview Question

Follow these steps to guide the discussion:

1.  **Outline use cases, constraints, and assumptions:** Gather requirements, scope the problem, ask clarifying questions.
2.  **Create a high-level design:** Sketch the main components and connections, and justify your ideas.
3.  **Design core components:** Dive into the details for each core component.
4.  **Scale the design:** Identify and address bottlenecks, given the constraints.

---

## System Design Interview Questions with Solutions

These are common system design interview questions, with sample discussions, code, and diagrams to help you master these topics:

*   [Design Pastebin.com (or Bit.ly)](#design-pastebincom-or-bitly)
*   [Design the Twitter Timeline and Search (or Facebook Feed and Search)](#design-the-twitter-timeline-and-search-or-facebook-feed-and-search)
*   [Design a Web Crawler](#design-a-web-crawler)
*   [Design Mint.com](#design-mintcom)
*   [Design the Data Structures for a Social Network](#design-the-data-structures-for-a-social-network)
*   [Design a Key-Value Store for a Search Engine](#design-a-key-value-store-for-a-search-engine)
*   [Design Amazon's Sales Ranking by Category Feature](#design-amazons-sales-ranking-by-category-feature)
*   [Design a System That Scales to Millions of Users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)

---
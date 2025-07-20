# System Design Primer: Your Comprehensive Guide to Building Scalable Systems

**Master system design concepts and ace your technical interviews with this open-source resource.**  [Explore the original repository](https://github.com/donnemartin/system-design-primer) for in-depth explanations, practical examples, and interview preparation materials.

## Key Features

*   **Organized Collection of Resources:** Learn how to design large-scale systems with an organized collection of resources found throughout the web.
*   **Interview Prep:** Prepare for system design interviews with common questions, sample solutions, and study guides.
*   **Community Driven:** Benefit from an open-source project, with contributions from the community that is constantly updated.
*   **Flashcard Decks:** Reinforce your learning with Anki flashcard decks for system design, system design exercises, and object-oriented design exercises.

## Table of Contents

*   [Motivation](#motivation)
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
*   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [System Design Topics: Start Here](#system-design-topics-start-here)
*   [Performance vs. Scalability](#performance-vs-scalability)
*   [Latency vs. Throughput](#latency-vs-throughput)
*   [Availability vs. Consistency](#availability-vs-consistency)
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
        *   [Master-slave replication](#master-slave-replication)
        *   [Master-master replication](#master-master-replication)
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
        *   [Write-behind (write-back)](#write-behind-write-back)
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

This repository provides a comprehensive learning experience for designing large-scale systems and preparing for system design interviews. The system design primer is an organized collection of resources to help you build systems at scale. It's a great way to learn from the community.

## Study Guide

Get started with the [study guide](#study-guide) to determine how you should best prepare for your interviews.

## How to Approach a System Design Interview Question

Follow the [step-by-step guide](#how-to-approach-a-system-design-interview-question) to approach system design questions in your interviews.

## System Design Interview Questions with Solutions

Explore common system design interview questions, with sample discussions, code, and diagrams.

*   Design Pastebin.com (or Bit.ly) ([Solution](solutions/system_design/pastebin/README.md))
*   Design the Twitter timeline and search (or Facebook feed and search) ([Solution](solutions/system_design/twitter/README.md))
*   Design a web crawler ([Solution](solutions/system_design/web_crawler/README.md))
*   Design Mint.com ([Solution](solutions/system_design/mint/README.md))
*   Design the data structures for a social network ([Solution](solutions/system_design/social_graph/README.md))
*   Design a key-value store for a search engine ([Solution](solutions/system_design/query_cache/README.md))
*   Design Amazon's sales ranking by category feature ([Solution](solutions/system_design/sales_rank/README.md))
*   Design a system that scales to millions of users on AWS ([Solution](solutions/system_design/scaling_aws/README.md))

## Object-Oriented Design Interview Questions with Solutions

*   Design a hash map ([Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb))
*   Design a least recently used cache ([Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb))
*   Design a call center ([Solution](solutions/object_oriented_design/call_center/call_center.ipynb))
*   Design a deck of cards ([Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb))
*   Design a parking lot ([Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb))
*   Design a chat server ([Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb))

---
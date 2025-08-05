# System Design Primer: Your Guide to Building Scalable Systems

**Master the art of designing large-scale systems and ace your system design interviews with this comprehensive, open-source resource!**  [Explore the original repository](https://github.com/donnemartin/system-design-primer) for a deeper dive.

## Key Features

*   **Comprehensive Coverage:** Explore a wide range of system design topics, from fundamental principles to advanced architectures.
*   **Interview Prep:** Get ready for system design interviews with detailed explanations, example questions, and sample solutions.
*   **Community Driven:** Benefit from a continually updated, open-source project with contributions from the community.
*   **Practical Resources:** Access Anki flashcards for spaced repetition and interactive coding challenges to solidify your knowledge.
*   **Real-World Examples:** Learn from architectures of successful companies and industry best practices.
*   **Multilingual Support:** Available in various languages, making it accessible to a global audience.

## Table of Contents

*   [System Design Topics: Start Here](#system-design-topics-start-here)
    *   [Performance vs. Scalability](#performance-vs-scalability)
    *   [Latency vs. Throughput](#latency-vs-throughput)
    *   [Availability vs. Consistency](#availability-vs-consistency)
        *   [CAP Theorem](#cap-theorem)
        *   [Consistency Patterns](#consistency-patterns)
        *   [Availability Patterns](#availability-patterns)
*   [Core System Design Concepts](#core-system-design-concepts)
    *   [Domain Name System](#domain-name-system)
    *   [Content Delivery Network (CDN)](#content-delivery-network)
    *   [Load Balancer](#load-balancer)
    *   [Reverse Proxy](#reverse-proxy-web-server)
    *   [Application Layer](#application-layer)
    *   [Database](#database)
    *   [Cache](#cache)
    *   [Asynchronism](#asynchronism)
    *   [Communication](#communication)
    *   [Security](#security)
*   [Interview Preparation](#interview-preparation)
    *   [Study Guide](#study-guide)
    *   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
    *   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
    *   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
    *   [Additional System Design Interview Questions](#additional-system-design-interview-questions)
*   [Appendix](#appendix)
    *   [Powers of Two Table](#powers-of-two-table)
    *   [Latency Numbers Every Programmer Should Know](#latency-numbers-every-programmer-should-know)
    *   [Real-World Architectures](#real-world-architectures)
    *   [Company Architectures](#company-architectures)
    *   [Company Engineering Blogs](#company-engineering-blogs)
*   [Contributing](#contributing)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

## System Design Topics: Start Here

Begin your journey by understanding the fundamental principles of system design.

### Performance vs. Scalability

*   A **scalable** service results in increased **performance** as resources are added.
*   Performance problems: System is slow for a *single* user.
*   Scalability problems: System is fast for a single user but slow under *heavy* load.

### Latency vs. Throughput

*   **Latency**: Time to perform an action.
*   **Throughput**: Actions per unit of time.
*   Goal: **Maximal throughput** with **acceptable latency**.

### Availability vs. Consistency

*   **Consistency**: Every read gets the most recent write or an error.
*   **Availability**: Every request gets a response.
*   **Partition Tolerance**: The system continues to operate despite network failures.
    *   You can only support two of these three guarantees due to the CAP theorem.

#### CAP Theorem

The CAP theorem is the cornerstone of designing distributed systems.

![CAP Theorem Diagram](images/bgLMI2u.png)

#### Consistency Patterns

*   **Weak Consistency**: Reads may or may not see the write.
*   **Eventual Consistency**: Reads will eventually see the write (typically milliseconds).
*   **Strong Consistency**: Reads will always see the most recent write.

#### Availability Patterns

*   **Fail-over**: The system should continue to operate despite the failure of one or more components.
    *   **Active-Passive**: One server is active, the other passive.
    *   **Active-Active**: Both servers manage traffic.
*   **Replication**: Copying data across multiple nodes to ensure data redundancy.

## Core System Design Concepts

Dive into key topics for building scalable systems.

### Domain Name System (DNS)

*   Translates domain names (e.g., `www.example.com`) to IP addresses.
*   Hierarchical, with authoritative servers and caching.

### Content Delivery Network (CDN)

*   Globally distributed network of proxy servers.
*   Serves content from locations closer to the user.
    *   **Push CDNs**: You control content and updates.
    *   **Pull CDNs**: Content pulled from your server on demand.

### Load Balancer

*   Distributes client requests to computing resources.
    *   Active-Passive
    *   Active-Active
    *   Layer 4 and Layer 7

### Reverse Proxy (Web Server)

*   Centralizes internal services.
*   Offers unified interfaces to the public.
*   Provides increased security, scalability, caching, and more.

### Application Layer

*   Separates the web tier from the application layer.
*   Allows for independent scaling.
    *   Microservices

### Database

*   A central component for storing and managing data.
    *   Relational Database Management System (RDBMS) - SQL
        *   Master-Slave and Master-Master Replication
        *   Federation
        *   Sharding
        *   Denormalization
    *   NoSQL
        *   Key-Value Stores
        *   Document Stores
        *   Wide Column Stores
        *   Graph Databases
    *   SQL or NoSQL
        *   Key differences in architecture.
        *   Consider your data's structure and access patterns.

### Cache

*   Improves performance and reduces load on servers/databases.
    *   Client Caching
    *   CDN Caching
    *   Web Server Caching
    *   Database Caching
    *   Application Caching
    *   Caching at the Database Query Level
    *   Caching at the Object Level
    *   Cache-Aside, Write-Through, Write-Behind, Refresh-Ahead

### Asynchronism

*   Improves response times for expensive operations.
    *   Message Queues
    *   Task Queues
    *   Back Pressure

### Communication

*   Protocols for transferring data between clients and servers.
    *   Hypertext Transfer Protocol (HTTP)
    *   Transmission Control Protocol (TCP)
    *   User Datagram Protocol (UDP)
    *   Remote Procedure Call (RPC)
    *   Representational State Transfer (REST)

### Security

*   Encryption, input sanitization, principle of least privilege.

## Interview Preparation

Prepare for your system design interview.

### Study Guide

*   Recommended study approach based on your interview timeline.
*   Prioritize breadth and depth based on your needs.

### How to Approach a System Design Interview Question

1.  **Outline use cases, constraints, and assumptions.**
2.  **Create a high-level design.**
3.  **Design core components.**
4.  **Scale the design.**
    *   Back-of-the-envelope calculations
        *   Powers of Two Table
        *   Latency numbers every programmer should know

### System Design Interview Questions with Solutions

*   Pastebin, Twitter timeline and search, Web Crawler, Mint.com, Data Structures for a Social Network, Key-Value Store, Amazon's Sales Ranking, Scaling AWS
    *   Example of Design Pastebin:

    ![Pastebin Design](images/4edXG0T.png)

### Object-Oriented Design Interview Questions with Solutions

*   Hash Map, Least Recently Used Cache, Call Center, Deck of Cards, Parking Lot, Chat Server

### Additional System Design Interview Questions

*   Design a file sync service, search engine, Google docs, key-value store, cache system, recommendation system, and more.

## Appendix

Essential resources for system design.

### Powers of Two Table

*   Quick reference for memory and storage sizes.

### Latency Numbers Every Programmer Should Know

*   Important latency values for various operations.
*   Quick reference.

### Real-World Architectures

*   Articles on real-world system designs.
    *   Distributed systems
    *   Databases
    *   File Systems
    *   Misc
    *   Focus on Shared principles and common technologies.

### Company Architectures

*   Architectural overviews from companies like Amazon, Netflix, and Uber.

### Company Engineering Blogs

*   Links to engineering blogs from top tech companies.

## Contributing

*   Fix errors, improve sections, add new sections, translate.
*   Review the [Contributing Guidelines](CONTRIBUTING.md).

## Credits

*   Acknowledgments to key contributors and resources.

## Contact Info

*   Contact information for the author.

## License

*   Creative Commons Attribution 4.0 International License (CC BY 4.0)
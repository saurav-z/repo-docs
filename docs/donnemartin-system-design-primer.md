# System Design Primer: Your Guide to Building Scalable Systems

**Learn how to design large-scale systems and ace your system design interview with this comprehensive, open-source guide.  Explore resources, practice questions, and solutions at the [original repo](https://github.com/donnemartin/system-design-primer).**

## Key Features

*   **Extensive Resources:** Organized collection of resources on system design principles.
*   **Interview Prep:** Guide to system design interview, with sample questions and solutions.
*   **Community-Driven:** An open-source project that thrives on contributions from the community.
*   **Anki Flashcards:** Spaced repetition decks for retaining key system design concepts.
*   **Coding Challenges:** Resources to help you prep for the coding interview.

## Topics Covered

*   **Scalability Principles:** Learn about performance vs. scalability, latency vs. throughput, and availability vs. consistency.
*   **System Design Topics:** Explore DNS, CDNs, load balancers, reverse proxies, application layers (microservices), databases (SQL/NoSQL), caching, asynchronism, and communication protocols (TCP/UDP, RPC, REST).
*   **Interview Preparation:** Study guides, how to approach system design questions, object-oriented design interview questions, and back-of-the-envelope calculations.
*   **Practical Solutions:** Detailed solutions to common system design interview questions (Pastebin, Twitter timeline, web crawler, and more).
*   **Real-World Architectures:** Dive into the design of systems like Google, Amazon, Facebook, and Netflix.
*   **Security:** Get an overview of key security considerations.

## Index of System Design Topics

### System Design Topics: Start Here
* Step 1: Review the scalability video lecture
* Step 2: Review the scalability article
* Next Steps

### Performance vs. Scalability

### Latency vs. Throughput

### Availability vs. Consistency

*   CAP Theorem
*   CP - Consistency and Partition Tolerance
*   AP - Availability and Partition Tolerance

### Consistency Patterns
*   Weak consistency
*   Eventual consistency
*   Strong consistency

### Availability Patterns
*   Fail-over
    *   Active-passive
    *   Active-active
*   Replication
    *   Master-slave
    *   Master-master
*   Availability in Numbers

### Domain Name System

### Content Delivery Network
*   Push CDNs
*   Pull CDNs

### Load Balancer
*   Active-passive
*   Active-active
*   Layer 4 load balancing
*   Layer 7 load balancing
*   Horizontal Scaling

### Reverse Proxy (web server)

### Application Layer
*   Microservices
*   Service Discovery

### Database
*   Relational Database Management System (RDBMS)
    *   Master-slave replication
    *   Master-master replication
    *   Federation
    *   Sharding
    *   Denormalization
    *   SQL Tuning
*   NoSQL
    *   Key-value store
    *   Document store
    *   Wide column store
    *   Graph Database
*   SQL or NoSQL

### Cache
*   Client caching
*   CDN caching
*   Web server caching
*   Database caching
*   Application caching
*   Caching at the database query level
*   Caching at the object level
*   When to update the cache
    *   Cache-aside
    *   Write-through
    *   Write-behind (write-back)
    *   Refresh-ahead

### Asynchronism
*   Message queues
*   Task queues
*   Back pressure

### Communication
*   Transmission Control Protocol (TCP)
*   User Datagram Protocol (UDP)
*   Remote Procedure Call (RPC)
*   Representational State Transfer (REST)

### Security

### Appendix
*   Powers of Two Table
*   Latency Numbers Every Programmer Should Know
*   Additional System Design Interview Questions
*   Real-World Architectures
*   Company Architectures
*   Company Engineering Blogs

### Study Guide

*   **Short Timeline**: Aim for Breadth
*   **Medium Timeline**: Aim for Breadth and Some Depth
*   **Long Timeline**: Aim for Breadth and More Depth

### How to Approach a System Design Interview Question
*   Step 1: Outline use cases, constraints, and assumptions
*   Step 2: Create a high-level design
*   Step 3: Design core components
*   Step 4: Scale the design
*   Back-of-the-Envelope Calculations

### System Design Interview Questions with Solutions
*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

### Object-oriented Design Interview Questions with Solutions

### Under Development
*   Distributed computing with MapReduce
*   Consistent Hashing
*   Scatter Gather

### Credits
### Contact Info
### License
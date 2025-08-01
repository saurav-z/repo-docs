# System Design Primer: Your Guide to Building Scalable Systems

**Master system design, ace your interviews, and design large-scale systems with this comprehensive open-source resource.**  [Explore the original repo](https://github.com/donnemartin/system-design-primer) for a deeper dive into these concepts and more!

## Key Features

*   **Comprehensive Coverage:** Learn the principles, patterns, and technologies behind designing scalable systems.
*   **Interview Prep:** Master system design interview questions with sample discussions, code, and diagrams.
*   **Open Source & Community Driven:** Benefit from a continually updated resource, with contributions from the community.
*   **Anki Flashcards:** Reinforce your learning with Anki flashcard decks for key concepts and exercises.
*   **Practical Exercises:**  Practice with example system design questions, including solutions for common systems like Pastebin, Twitter, and more.
*   **Resources & References:** Access valuable resources, including real-world architectures, company engineering blogs, and back-of-the-envelope calculation guides.

## Introduction

This repository serves as a comprehensive guide to system design, providing the knowledge and resources you need to build scalable systems and prepare for system design interviews. Whether you're a software engineer aiming to understand system design principles or a candidate prepping for a technical interview, this primer offers valuable insights.

## Core Concepts

### System Design Fundamentals

*   **Scalability vs. Performance:** Understand the key differences and how to optimize for each.
*   **Latency vs. Throughput:** Learn how to balance these crucial metrics for optimal system performance.
*   **CAP Theorem:**  Explore the trade-offs between Consistency, Availability, and Partition tolerance in distributed systems.
*   **Consistency Patterns:**  Examine Weak, Eventual, and Strong consistency models.
*   **Availability Patterns:**  Discover fail-over, replication, and availability in numbers.

### Essential Components

*   **DNS:** Domain Name System and its function in internet architecture.
*   **CDN:** Content Delivery Networks, how they work, and their benefits.
*   **Load Balancers:** Understand the importance of balancing and distributing network traffic
*   **Reverse Proxies:** Explore how they improve performance and security.
*   **Application Layer:** How the application layer functions with microservices.

### Databases and Caching

*   **Databases:**
    *   Relational Databases (RDBMS)
        *   Master-slave replication
        *   Master-master replication
        *   Federation
        *   Sharding
        *   Denormalization
        *   SQL tuning
    *   NoSQL Databases
        *   Key-value stores
        *   Document stores
        *   Wide-column stores
        *   Graph Databases
    *   SQL vs. NoSQL: Understanding the differences and trade-offs.
*   **Caching:**  Learn various caching strategies to optimize system performance.

### Asynchronism and Communication

*   **Asynchronism:**
    *   Message queues
    *   Task queues
    *   Back pressure
*   **Communication:**
    *   TCP, UDP, RPC, REST - Learn how these communication protocols work.

### Security

*   Important aspects of API security.

## Study Guide

A suggested study guide is provided based on interview timeline (short, medium, long) and experience levels.

## System Design Interview Preparation

This section guides you through:

*   **How to Approach a System Design Interview Question:** A step-by-step guide to tackling system design questions.
*   **System Design Interview Questions with Solutions:**
    *   Design Pastebin.com (or Bit.ly)
    *   Design the Twitter timeline and search (or Facebook feed and search)
    *   Design a web crawler
    *   Design Mint.com
    *   Design the data structures for a social network
    *   Design a key-value store for a search engine
    *   Design Amazon's sales ranking by category feature
    *   Design a system that scales to millions of users on AWS

*   **Object-oriented design interview questions with solutions:**  This section is currently under development.

## Appendix

*   Powers of two table.
*   Latency numbers every programmer should know.
*   Additional system design interview questions.
*   Real-world architectures.
*   Company architectures.
*   Company engineering blogs.

## Contributing

This is a community-driven, open-source project! Contributions are welcome!

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
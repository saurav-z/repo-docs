# System Design Primer: Your Guide to Building Scalable Systems

**Master the art of designing large-scale systems and ace your system design interviews with this comprehensive and community-driven resource. [Explore the original repo](https://github.com/donnemartin/system-design-primer) for a deeper dive.**

This README provides a structured overview of system design concepts, along with resources to help you prepare for system design interviews and build robust, scalable systems.

## Key Features

*   **Comprehensive Coverage**: From fundamental principles to advanced architectures, this primer covers a wide range of system design topics.
*   **Community-Driven**: Benefit from a continually updated, open-source project with contributions from a global community of engineers.
*   **Interview Preparation**: Prepare for technical interviews with detailed solutions to common system design questions.
*   **Anki Flashcards**: Solidify your knowledge with ready-to-use Anki flashcard decks for on-the-go learning.
*   **Interactive Coding Challenges**: Prepare for coding interviews with resources from sister repo [**Interactive Coding Challenges**](https://github.com/donnemartin/interactive-coding-challenges) with associated Anki decks.

## Core Concepts

*   **System Design Topics**: Explore a broad spectrum of topics, beginning with:
    *   Performance vs. Scalability
    *   Latency vs. Throughput
    *   Availability vs. Consistency (CAP Theorem)
*   **Consistency Patterns**: Understand different approaches to data consistency, including:
    *   Weak Consistency
    *   Eventual Consistency
    *   Strong Consistency
*   **Availability Patterns**: Explore strategies for ensuring system availability:
    *   Fail-over (Active-Passive, Active-Active)
    *   Replication
*   **Infrastructure Components**: Delve into crucial system components:
    *   Domain Name System (DNS)
    *   Content Delivery Networks (CDNs)
    *   Load Balancers
    *   Reverse Proxies
    *   Application Layer (Microservices, Service Discovery)
*   **Data Management**: Learn about various database technologies:
    *   Relational Database Management Systems (RDBMS)
    *   NoSQL Databases (Key-Value, Document, Wide Column, Graph)
*   **Caching Strategies**: Master the art of caching for performance optimization.
    *   Client Caching
    *   CDN Caching
    *   Web Server Caching
    *   Database Caching
    *   Application Caching
*   **Asynchronous Processing**: Leverage asynchronous techniques for improved performance:
    *   Message Queues
    *   Task Queues
    *   Back Pressure
*   **Communication Protocols**: Understand the essentials of communication:
    *   HTTP, TCP, UDP, RPC, REST
*   **Security Basics**: Understand basic security best practices.
    *   Encryption, Input Sanitization, Principle of Least Privilege

## Study Guide

This primer offers a flexible study guide to help you structure your learning based on your interview timeline.

*   **Short Timeline**: Focus on breadth, reviewing key system design topics, architectural articles, and a few interview questions.
*   **Medium Timeline**: Aim for both breadth and depth, practicing with a wider range of interview questions.
*   **Long Timeline**: Achieve deeper knowledge, and complete more practice questions.

## How to Approach a System Design Interview Question

Master a systematic approach to system design questions.

1.  **Outline Use Cases, Constraints, and Assumptions**: Define the problem scope.
2.  **Create a High-Level Design**: Sketch the main components and their connections.
3.  **Design Core Components**: Dive into details for each core component.
4.  **Scale the Design**: Identify and address bottlenecks.

## System Design Interview Questions with Solutions

Practice and learn from example solutions. Solutions are in the `solutions/` folder.

*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

## Object-Oriented Design Interview Questions with Solutions

Explore common object-oriented design questions.

*   Design a hash map
*   Design a least recently used cache
*   Design a call center
*   Design a deck of cards
*   Design a parking lot
*   Design a chat server

## Appendix: Essential References

*   **Powers of Two Table**: Quickly estimate storage and memory requirements.
*   **Latency Numbers Every Programmer Should Know**: Understand performance trade-offs.
*   **Additional System Design Interview Questions**: Expand your practice with additional challenges.
*   **Real-World Architectures**: Study successful system designs.
*   **Company Engineering Blogs**: Discover how companies approach their designs.

## Contributing

Help improve this resource by:

*   Fixing errors.
*   Improving sections.
*   Adding new sections.
*   Translating the guide (TRANSLATIONS.md).

## Contact & License

Reach out with any questions. The code and resources are available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
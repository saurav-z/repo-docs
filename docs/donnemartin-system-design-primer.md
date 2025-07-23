# System Design Primer: Your Guide to Building Scalable Systems

**Want to ace your system design interview and build robust, scalable applications?** This comprehensive guide provides a structured and organized collection of resources to help you master the art of system design. Check out the original repo [here](https://github.com/donnemartin/system-design-primer).

## Key Features:

*   **Comprehensive Coverage:** Learn about the fundamental principles of system design, including performance, scalability, latency, consistency, and availability.
*   **In-depth Topics:** Explore key areas like Domain Name System (DNS), Content Delivery Networks (CDNs), Load Balancers, Caching, Databases (SQL, NoSQL), Message Queues, and more.
*   **Interview Prep:** Prepare for system design interviews with a structured study guide, sample questions, and solutions, including object-oriented design and additional resources.
*   **Community-Driven:** Benefit from an open-source project where you can contribute to fix errors, improve sections, or add new ones.
*   **Practical Resources:** Utilize Anki flashcards for spaced repetition learning and access to interactive coding challenges.

## Table of Contents

*   [Introduction](#the-system-design-primer)
*   [Key Features](#key-features)
*   [Table of Contents](#table-of-contents)
*   [What is System Design?](#motivation)
    *   [How to design large-scale systems](#learn-how-to-design-large-scale-systems)
    *   [The power of the open source community](#learn-from-the-open-source-community)
    *   [Interview Preparation](#prep-for-the-system-design-interview)
*   [Anki Flashcards](#anki-flashcards)
    *   [Interactive Coding Challenges](#coding-resource-interactive-coding-challenges)
*   [Contribute](#contributing)
*   [Index of System Design Topics](#index-of-system-design-topics)
    *   [System design topics: start here](#system-design-topics-start-here)
    *   [Performance vs. Scalability](#performance-vs-scalability)
    *   [Latency vs. Throughput](#latency-vs-throughput)
    *   [Availability vs. Consistency](#availability-vs-consistency)
    *   [Consistency Patterns](#consistency-patterns)
    *   [Availability Patterns](#availability-patterns)
    *   [Domain Name System](#domain-name-system)
    *   [Content Delivery Network](#content-delivery-network)
    *   [Load Balancer](#load-balancer)
    *   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Application Layer](#application-layer)
    *   [Database](#database)
    *   [Cache](#cache)
    *   [Asynchronism](#asynchronism)
    *   [Communication](#communication)
    *   [Security](#security)
    *   [Appendix](#appendix)
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
*   [Consistency Patterns](#consistency-patterns)
*   [Availability Patterns](#availability-patterns)
*   [Domain Name System](#domain-name-system)
*   [Content Delivery Network](#content-delivery-network)
*   [Load Balancer](#load-balancer)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
*   [Application Layer](#application-layer)
*   [Database](#database)
*   [Cache](#cache)
*   [Asynchronism](#asynchronism)
*   [Communication](#communication)
*   [Security](#security)
*   [Appendix](#appendix)
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
*   [Object-Oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [Real World Architectures](#real-world-architectures)
*   [Company Architectures](#company-architectures)
*   [Company Engineering Blogs](#company-engineering-blogs)
*   [Under Development](#under-development)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

## What is System Design?

System design is a broad topic with vast resources. This repository serves as an organized collection of resources to help you learn how to build systems at scale.

### How to design large-scale systems

Learning how to design scalable systems will help you become a better engineer.

### The power of the open source community

This is a continually updated, open-source project.

### Interview Preparation

In addition to coding interviews, system design is a **required component** of the **technical interview process** at many tech companies.

*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions, **with solutions**](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions, **with solutions**](#object-oriented-design-interview-questions-with-solutions)
*   [Additional system design interview questions](#additional-system-design-interview-questions)

## Anki Flashcards

The provided [Anki flashcard decks](https://apps.ankiweb.net/) use spaced repetition to help you retain key system design concepts.

*   [System design deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design.apkg)
*   [System design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design%20Exercises.apkg)
*   [Object oriented design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/OO%20Design.apkg)

## Interactive Coding Challenges

Looking for resources to help you prep for the [**Coding Interview**](https://github.com/donnemartin/interactive-coding-challenges)?

Check out the sister repo [**Interactive Coding Challenges**](https://github.com/donnemartin/interactive-coding-challenges), which contains an additional Anki deck:

*   [Coding deck](https://github.com/donnemartin/interactive-coding-challenges/tree/master/anki_cards/Coding.apkg)

## Contribute

Feel free to submit pull requests to help:

*   Fix errors
*   Improve sections
*   Add new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

Content that needs some polishing is placed [under development](#under-development).

Review the [Contributing Guidelines](CONTRIBUTING.md).
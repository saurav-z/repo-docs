# System Design Primer: Your Comprehensive Guide to Designing Large-Scale Systems

**Tackle system design interviews and build scalable applications with this organized collection of resources. [Explore the original repo](https://github.com/donnemartin/system-design-primer).**

## Key Features

*   **Structured Learning:** An organized collection of resources covering core system design principles.
*   **Interview Prep:** Guides for system design interviews, including common questions, solutions, and study resources.
*   **Community Driven:** A continually updated, open-source project with contributions from the community.
*   **Extensive Coverage:** Detailed explanations of crucial topics such as load balancing, databases, caching, and more.
*   **Practical Resources:** Includes Anki flashcards and links to valuable company architecture resources and engineering blogs.

## 1. Motivation: Master the Fundamentals of System Design

> Learn how to design large-scale systems and ace your system design interview.

*   **Why System Design Matters:** Designing scalable systems is essential for becoming a better engineer.
*   **Comprehensive Resources:** This repository provides an organized collection of resources on system design principles.
*   **Open-Source Community:** Benefit from a continually updated, open-source project with community contributions.
*   **Interview Preparation:** System design is a crucial part of technical interviews, especially for senior roles.

### 1.1. Core Concepts: Building Blocks of Scalable Systems

*   **[Index of system design topics](#index-of-system-design-topics):** Start here for a broad understanding of system design.
    *   **[Performance vs scalability](#performance-vs-scalability)**
    *   **[Latency vs throughput](#latency-vs-throughput)**
    *   **[Availability vs consistency](#availability-vs-consistency)**
        *   **[CAP theorem](#cap-theorem)**
    *   **[Consistency patterns](#consistency-patterns)**
    *   **[Availability patterns](#availability-patterns)**
    *   **[Domain name system](#domain-name-system)**
    *   **[Content delivery network](#content-delivery-network)**
    *   **[Load balancer](#load-balancer)**
    *   **[Reverse proxy (web server)](#reverse-proxy-web-server)**
    *   **[Application layer](#application-layer)**
    *   **[Database](#database)**
    *   **[Cache](#cache)**
    *   **[Asynchronism](#asynchronism)**
    *   **[Communication](#communication)**
    *   **[Security](#security)**
    *   **[Appendix](#appendix)**

### 1.2.  Interview Prep:  Ace the System Design Interview

*   **[Study guide](#study-guide):** Optimize your preparation based on your interview timeline (short, medium, long).
*   **[How to approach a system design interview question](#how-to-approach-a-system-design-interview-question):** Follow the structured steps to effectively address design questions.
*   **[System design interview questions with solutions](#system-design-interview-questions-with-solutions):** Practice with common questions, solutions, and diagrams.
*   **[Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions):** A valuable section for the development of system design skills
*   **[Additional system design interview questions](#additional-system-design-interview-questions):** Explore a wider range of practice questions.

### 1.3. Practical Tools: Enhance Your Learning Experience

*   **[Anki flashcards](#anki-flashcards):** Use spaced repetition to memorize key concepts.
    *   [System design deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design.apkg)
    *   [System design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design%20Exercises.apkg)
    *   [Object oriented design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/OO%20Design.apkg)
*   [Coding Interview Challenges](https://github.com/donnemartin/interactive-coding-challenges): Complementary resources for coding interview preparation.

## 2.  Contributing: Become a Part of the Community

> Contribute to the project and help improve the guide by submitting pull requests.

*   Fix errors, improve sections, and add new content.
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28) the guide into multiple languages.

## 3. Index of System Design Topics: Dive into the Details

>  Summaries of various system design topics, including pros and cons.

*   **[System design topics: start here](#system-design-topics-start-here):** Kickstart your system design journey.
    *   **[Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)**
    *   **[Step 2: Review the scalability article](#step-2-review-the-scalability-article)**
    *   **[Next steps](#next-steps)**
*   **[Performance vs scalability](#performance-vs-scalability)**
*   **[Latency vs throughput](#latency-vs-throughput)**
*   **[Availability vs consistency](#availability-vs-consistency)**
    *   **[CAP theorem](#cap-theorem)**
*   **[Consistency patterns](#consistency-patterns)**
*   **[Availability patterns](#availability-patterns)**
*   **[Domain name system](#domain-name-system)**
*   **[Content delivery network](#content-delivery-network)**
*   **[Load balancer](#load-balancer)**
*   **[Reverse proxy (web server)](#reverse-proxy-web-server)**
*   **[Application layer](#application-layer)**
*   **[Database](#database)**
*   **[Cache](#cache)**
*   **[Asynchronism](#asynchronism)**
*   **[Communication](#communication)**
*   **[Security](#security)**
*   **[Appendix](#appendix)**
    *   **[Powers of two table](#powers-of-two-table)**
    *   **[Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)**
    *   **[Additional system design interview questions](#additional-system-design-interview-questions)**
    *   **[Real world architectures](#real-world-architectures)**
    *   **[Company architectures](#company-architectures)**
    *   **[Company engineering blogs](#company-engineering-blogs)**
*   **[Under development](#under-development)**

## 4.  Study Guide: Tailor Your Preparation

>  Follow the suggested topics to review based on your interview timeline (short, medium, long).

|                   | Short | Medium | Long |
| ----------------- | ----- | ------ | ---- |
| Broad Understanding | :+1:  | :+1:   | :+1:  |
| Company Blogs     | :+1:  | :+1:   | :+1:  |
| Real World        | :+1:  | :+1:   | :+1:  |
| Interview Approach| :+1:  | :+1:   | :+1:  |
| Practice          | Some  | Many   | Most |
| Object-Oriented   | Some  | Many   | Most |
| More Questions    | Some  | Many   | Most |

## 5.  How to Approach a System Design Interview Question: Master the Process

>  Follow these steps to guide your discussion.

1.  **Outline use cases, constraints, and assumptions.**
2.  **Create a high-level design.**
3.  **Design core components.**
4.  **Scale the design.**
    *   Back-of-the-envelope calculations
    *   Refer to the Appendix

## 6. System Design Interview Questions with Solutions: Practice Makes Perfect

>  Common system design interview questions with sample discussions, code, and diagrams.

*   **Design Pastebin.com (or Bit.ly)**
*   **Design the Twitter timeline and search (or Facebook feed and search)**
*   **Design a web crawler**
*   **Design Mint.com**
*   **Design the data structures for a social network**
*   **Design a key-value store for a search engine**
*   **Design Amazon's sales ranking by category feature**
*   **Design a system that scales to millions of users on AWS**

## 7. Object-oriented design interview questions with solutions:

>  Common object-oriented design interview questions with sample discussions, code, and diagrams.

*   **Design a hash map**
*   **Design a least recently used cache**
*   **Design a call center**
*   **Design a deck of cards**
*   **Design a parking lot**
*   **Design a chat server**
*   **Design a circular array**

## 8. System Design Topics: Start Here:

> New to system design?

### 8.1. Review the scalability video lecture

[Scalability Lecture at Harvard](https://www.youtube.com/watch?v=-W9F__D3oY4)

* Topics covered:
    * Vertical scaling
    * Horizontal scaling
    * Caching
    * Load balancing
    * Database replication
    * Database partitioning

### 8.2. Review the scalability article

[Scalability](https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono)

* Topics covered:
    * [Clones](https://web.archive.org/web/20220530193911/https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones)
    * [Databases](https://web.archive.org/web/20220602114024/https://www.lecloud.net/post/7994751381/scalability-for-dummies-part-2-database)
    * [Caches](https://web.archive.org/web/20230126233752/https://www.lecloud.net/post/9246290032/scalability-for-dummies-part-3-cache)
    * [Asynchronism](https://web.archive.org/web/20220926171507/https://www.lecloud.net/post/9699762917/scalability-for-dummies-part-4-asynchronism)

### 8.3. Next steps

Next, we'll look at high-level trade-offs:

*   **Performance** vs **scalability**
*   **Latency** vs **throughput**
*   **Availability** vs **consistency**

Keep in mind that **everything is a trade-off**.

Then we'll dive into more specific topics such as DNS, CDNs, and load balancers.

## 9. Appendix:

*   **[Powers of two table](#powers-of-two-table)**
*   **[Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)**
*   **[Additional system design interview questions](#additional-system-design-interview-questions)**
*   **[Real world architectures](#real-world-architectures)**
*   **[Company architectures](#company-architectures)**
*   **[Company engineering blogs](#company-engineering-blogs)**

## 10. Credits

*   [Hired in tech](http://www.hiredintech.com/system-design/the-system-design-process/)
*   [Cracking the coding interview](https://www.amazon.com/dp/0984782850/)
*   [High scalability](http://highscalability.com/)
*   [checkcheckzz/system-design-interview](https://github.com/checkcheckzz/system-design-interview)
*   [shashank88/system_design](https://github.com/shashank88/system_design)
*   [mmcgrana/services-engineering](https://github.com/mmcgrana/services-engineering)
*   [System design cheat sheet](https://gist.github.com/vasanthk/485d1c25737e8e72759f)
*   [A distributed systems reading list](http://dancres.github.io/Pages/)
*   [Cracking the system design interview](http://www.puncsky.com/blog/2016-02-13-crack-the-system-design-interview)

## 11. Contact info

Feel free to contact to discuss any issues, questions, or comments.

My contact info can be found on my [GitHub page](https://github.com/donnemartin).

## 12. License

*I am providing code and resources in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code and resources is from me and not my employer (Facebook).*

    Copyright 2017 Donne Martin

    Creative Commons Attribution 4.0 International License (CC BY 4.0)

    http://creativecommons.org/licenses/by/4.0/
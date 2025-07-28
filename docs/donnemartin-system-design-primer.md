# System Design Primer: Your Guide to Designing Large-Scale Systems

**Master the art of system design with this comprehensive, open-source guide to building scalable systems.**

[View the original repository](https://github.com/donnemartin/system-design-primer)

This repository offers a well-organized collection of resources and interview preparation materials to help you excel in system design interviews and build robust, scalable systems.

**Key Features:**

*   **Comprehensive Topic Coverage:** Delve into key system design topics, from fundamental principles to advanced architectures.
*   **Practical Solutions:** Study common system design interview questions complete with detailed discussions, code, and diagrams.
*   **Community-Driven Content:** Benefit from an active and collaborative community, ensuring the guide stays up-to-date and relevant.
*   **Interview Prep:** Includes a study guide, practice questions, and object-oriented design exercises.
*   **Anki Flashcards:** Utilize spaced repetition for enhanced retention of system design concepts with decks for system design, exercises, and object-oriented design.
*   **Language Support:** Available in English, Japanese, Simplified Chinese, Traditional Chinese, and multiple others.

**Sections Include:**

*   [Index of System Design Topics](#index-of-system-design-topics)
*   [Study Guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions with solutions](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)
*   [Additional system design interview questions](#additional-system-design-interview-questions)
*   [Real World Architectures](#real-world-architectures)
*   [Company Architectures](#company-architectures)
*   [Company Engineering Blogs](#company-engineering-blogs)

---

**Ready to contribute?**  All contributions are welcome!  Feel free to submit pull requests to help:

*   Fix errors
*   Improve sections
*   Add new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

---

## Index of System Design Topics

*   [System design topics: start here](#system-design-topics-start-here)
*   [Performance vs scalability](#performance-vs-scalability)
*   [Latency vs throughput](#latency-vs-throughput)
*   [Availability vs consistency](#availability-vs-consistency)
*   [Consistency patterns](#consistency-patterns)
*   [Availability patterns](#availability-patterns)
*   [Domain name system](#domain-name-system)
*   [Content delivery network](#content-delivery-network)
*   [Load balancer](#load-balancer)
*   [Reverse proxy (web server)](#reverse-proxy-web-server)
*   [Application layer](#application-layer)
*   [Database](#database)
*   [Cache](#cache)
*   [Asynchronism](#asynchronism)
*   [Communication](#communication)
*   [Security](#security)
*   [Appendix](#appendix)

---

## Study Guide

This section suggests topics to review based on your interview timeline.

| | Short | Medium | Long |
|---|---|---|---|
| Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work | :+1: | :+1: | :+1: |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | :+1: | :+1: | :+1: |
| Read through a few [Real world architectures](#real-world-architectures) | :+1: | :+1: | :+1: |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | :+1: | :+1: | :+1: |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Some | Many | Most |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Some | Many | Most |
| Review [Additional system design interview questions](#additional-system-design-interview-questions) | Some | Many | Most |

---

## How to Approach a System Design Interview Question

1.  **Outline Use Cases, Constraints, and Assumptions:** Define requirements and scope.
2.  **Create a High-Level Design:** Sketch components and connections.
3.  **Design Core Components:** Dive into details for each component.
4.  **Scale the Design:** Address bottlenecks using scalability principles.
5.  **Back-of-the-Envelope Calculations:** Use estimations to validate designs.

---

## System Design Interview Questions with Solutions

*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

---

## Object-Oriented Design Interview Questions with Solutions

*   Design a hash map
*   Design a least recently used cache
*   Design a call center
*   Design a deck of cards
*   Design a parking lot
*   Design a chat server

---

## Additional System Design Interview Questions

*   Design a file sync service like Dropbox
*   Design a search engine like Google
*   Design a scalable web crawler like Google
*   Design Google docs
*   Design a key-value store like Redis
*   Design a cache system like Memcached
*   Design a recommendation system like Amazon's
*   Design a tinyurl system like Bitly
*   Design a chat app like WhatsApp
*   Design a picture sharing system like Instagram
*   Design the Facebook news feed function
*   Design the Facebook timeline function
*   Design the Facebook chat function
*   Design a graph search function like Facebook's
*   Design a content delivery network like CloudFlare
*   Design a trending topic system like Twitter's
*   Design a random ID generation system
*   Return the top k requests during a time interval
*   Design a system that serves data from multiple data centers
*   Design an online multiplayer card game
*   Design a garbage collection system
*   Design an API rate limiter
*   Design a Stock Exchange (like NASDAQ or Binance)

---

## Real World Architectures

*   Data Processing
    *   MapReduce
    *   Spark
    *   Storm
*   Data Store
    *   Bigtable
    *   HBase
    *   Cassandra
    *   DynamoDB
    *   MongoDB
    *   Spanner
    *   Memcached
    *   Redis
*   File System
    *   Google File System (GFS)
    *   Hadoop File System (HDFS)
*   Misc
    *   Chubby
    *   Dapper
    *   Kafka
    *   Zookeeper

---

## Company Architectures

*   Amazon
*   Cinchcast
*   DataSift
*   Dropbox
*   ESPN
*   Google
*   Instagram
*   Justin.tv
*   Facebook
*   Flickr
*   Mailbox
*   Netflix
*   Pinterest
*   Playfish
*   PlentyOfFish
*   Salesforce
*   Stack Overflow
*   TripAdvisor
*   Tumblr
*   Twitter
*   Uber
*   WhatsApp
*   YouTube

---

## Company Engineering Blogs

*   Airbnb
*   Atlassian
*   AWS
*   Bitly
*   Box
*   Cloudera
*   Dropbox
*   Quora
*   Ebay
*   Evernote
*   Etsy
*   Facebook
*   Flickr
*   Foursquare
*   Github
*   Google
*   Groupon
*   Heroku
*   Hubspot
*   High Scalability
*   Instagram
*   Intel
*   Jane Street
*   LinkedIn
*   Microsoft
*   Netflix
*   Paypal
*   Pinterest
*   Reddit
*   Salesforce
*   Slack
*   Spotify
*   Stripe
*   Twilio
*   Twitter
*   Uber
*   Yahoo
*   Yelp
*   Zynga

---
# Python Cheat Sheet: Your Comprehensive Guide 🔗

**Get up to speed with Python quickly!** This cheat sheet provides a concise and SEO-optimized overview of Python's core features, from basic syntax to advanced libraries, perfect for both beginners and experienced developers. Explore Python's full potential with this easy-to-navigate resource, and find a wealth of information that helps you write effective, readable, and efficient Python code. Dive in and start coding with confidence!

[Original Repository](https://github.com/gto76/python-cheatsheet)

**Key Features:**

*   **Comprehensive Coverage:** Includes everything from fundamental data structures to advanced concepts like threading and web app development.
*   **Code Examples:** Clear and concise code snippets illustrate each concept, making it easy to understand and apply.
*   **Organized Structure:** Topics are logically grouped, allowing you to quickly find what you need.
*   **SEO Optimized:** Targeted keywords to help you quickly find this useful resource.

**Table of Contents:**

*   1.  [Collections](#collections) 🔗
*   2.  [Types](#types) 🔗
*   3.  [Syntax](#syntax) 🔗
*   4.  [System](#system) 🔗
*   5.  [Data](#data) 🔗
*   6.  [Advanced](#advanced) 🔗
*   7.  [Libraries](#libraries) 🔗
*   8.  [Multimedia](#multimedia) 🔗
*   9.  [Appendix](#appendix) 🔗
    *   [Cython](#cython) 🔗
    *   [Virtual Environments](#virtual-environments) 🔗
    *   [Basic Script Template](#basic-script-template) 🔗
    *   [Index](#index) 🔗

## 1. Collections 🔗

*   [List](#list) 🔗
*   [Dictionary](#dictionary) 🔗
*   [Set](#set) 🔗
*   [Tuple](#tuple) 🔗
*   [Range](#range) 🔗
*   [Enumerate](#enumerate) 🔗
*   [Iterator](#iterator) 🔗
*   [Generator](#generator) 🔗

## 2. Types 🔗

*   [Type](#type) 🔗
*   [String](#string) 🔗
*   [Regex](#regex) 🔗
*   [Format](#format) 🔗
*   [Numbers](#numbers) 🔗
*   [Combinatorics](#combinatorics) 🔗
*   [Datetime](#datetime) 🔗

## 3. Syntax 🔗

*   [Function](#function) 🔗
*   [Splat Operator](#splat-operator) 🔗
*   [Inline](#inline) 🔗
    *   [Lambda](#lambda) 🔗
    *   [Comprehensions](#comprehensions) 🔗
    *   [Map, Filter, Reduce](#map-filter-reduce) 🔗
    *   [Any, All](#any-all) 🔗
    *   [Conditional Expression](#conditional-expression) 🔗
    *   [And, Or](#and-or) 🔗
    *   [Walrus Operator](#walrus-operator) 🔗
    *   [Named Tuple, Enum, Dataclass](#named-tuple-enum-dataclass) 🔗
*   [Imports](#imports) 🔗
*   [Closure](#closure) 🔗
    *   [Partial](#partial) 🔗
    *   [Non-Local](#non-local) 🔗
*   [Decorator](#decorator) 🔗
*   [Class](#class) 🔗
    *   [Subclass](#subclass) 🔗
    *   [Type Annotations](#type-annotations) 🔗
    *   [Dataclass](#dataclass) 🔗
    *   [Property](#property) 🔗
    *   [Slots](#slots) 🔗
    *   [Copy](#copy) 🔗
*   [Duck Types](#duck-types) 🔗
    *   [Comparable](#comparable) 🔗
    *   [Hashable](#hashable) 🔗
    *   [Sortable](#sortable) 🔗
    *   [Iterator](#iterator) 🔗
    *   [Callable](#callable) 🔗
    *   [Context Manager](#context-manager) 🔗
*   [Iterable Duck Types](#iterable-duck-types) 🔗
    *   [Iterable](#iterable) 🔗
    *   [Collection](#collection) 🔗
    *   [Sequence](#sequence) 🔗
    *   [ABC Sequence](#abc-sequence) 🔗
*   [Enum](#enum) 🔗
*   [Exceptions](#exceptions) 🔗

## 4. System 🔗

*   [Exit](#exit) 🔗
*   [Print](#print) 🔗
*   [Input](#input) 🔗
*   [Command Line Arguments](#command-line-arguments) 🔗
*   [Open](#open) 🔗
*   [Paths](#paths) 🔗
*   [OS Commands](#os-commands) 🔗

## 5. Data 🔗

*   [JSON](#json) 🔗
*   [Pickle](#pickle) 🔗
*   [CSV](#csv) 🔗
*   [SQLite](#sqlite) 🔗
*   [Bytes](#bytes) 🔗
*   [Struct](#struct) 🔗
*   [Array](#array) 🔗
*   [Memory View](#memory-view) 🔗
*   [Deque](#deque) 🔗

## 6. Advanced 🔗

*   [Operator](#operator) 🔗
*   [Match Statement](#match-statement) 🔗
*   [Logging](#logging) 🔗
*   [Introspection](#introspection) 🔗
*   [Threading](#threading) 🔗
*   [Coroutines](#coroutines) 🔗

## 7. Libraries 🔗

*   [Progress Bar](#progress-bar) 🔗
*   [Plot](#plot) 🔗
*   [Table](#table) 🔗
*   [Console App](#console-app) 🔗
*   [GUI App](#gui-app) 🔗
*   [Scraping](#scraping) 🔗
    *   [Selenium](#selenium) 🔗
    *   [XPath](#xpath) 🔗
*   [Web App](#web-app) 🔗
*   [Profiling](#profiling) 🔗
    *   [Timing a Snippet](#timing-a-snippet) 🔗
    *   [Profiling by Line](#profiling-by-line) 🔗
    *   [Call and Flame Graphs](#call-and-flame-graphs) 🔗
    *   [Sampling and Memory Profilers](#sampling-and-memory-profilers) 🔗

## 8. Multimedia 🔗

*   [NumPy](#numpy) 🔗
    *   [Indexing](#indexing) 🔗
    *   [Broadcasting](#broadcasting) 🔗
    *   [Example](#example) 🔗
*   [Image](#image) 🔗
    *   [Modes](#modes) 🔗
    *   [Examples](#examples) 🔗
    *   [Image Draw](#image-draw) 🔗
*   [Animation](#animation) 🔗
*   [Audio](#audio) 🔗
    *   [Sample Values](#sample-values) 🔗
    *   [Read Float Samples from WAV File](#read-float-samples-from-wav-file) 🔗
    *   [Write Float Samples to WAV File](#write-float-samples-to-wav-file) 🔗
    *   [Examples](#examples) 🔗
    *   [Text to Speech](#text-to-speech) 🔗
*   [Synthesizer](#synthesizer) 🔗
*   [Pygame](#pygame) 🔗
    *   [Rect](#rect) 🔗
    *   [Surface](#surface) 🔗
    *   [Basic Mario Brothers Example](#basic-mario-brothers-example) 🔗
*   [Pandas](#pandas) 🔗
    *   [Series](#series) 🔗
        *   [Series — Aggregate, Transform, Map](#series--aggregate-transform-map) 🔗
    *   [DataFrame](#dataframe) 🔗
        *   [DataFrame — Merge, Join, Concat](#dataframe--merge-join-concat) 🔗
        *   [DataFrame — Aggregate, Transform, Map](#dataframe--aggregate-transform-map) 🔗
    *   [Multi-Index](#multi-index) 🔗
    *   [File Formats](#file-formats) 🔗
    *   [GroupBy](#groupby) 🔗
    *   [Rolling](#rolling) 🔗
*   [Plotly](#plotly) 🔗
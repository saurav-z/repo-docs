# Comprehensive Python Cheatsheet 🔗

> Master Python with this comprehensive cheatsheet, your go-to resource for quick reference and efficient coding, linking back to the original [GitHub repository](https://github.com/gto76/python-cheatsheet).

This cheatsheet provides a concise overview of essential Python concepts, from fundamental data structures to advanced libraries, designed to boost your productivity and understanding of the language.

## Key Features:

*   **Collections**: Lists, Dictionaries, Sets, Tuples, Ranges, Iterators, and Generators.
*   **Types**: Strings, Regular Expressions, Formatting, Numbers, Datetime, and more.
*   **Syntax**: Functions, Inlines, Imports, Decorators, Classes, Exceptions and more.
*   **System**: File I/O, Path Manipulation, OS Commands, and Command-Line Arguments.
*   **Data**: JSON, Pickle, CSV, SQLite, Bytes, and Struct.
*   **Advanced**: Operator Overloading, Match Statement, Logging, Introspection, and Threading.
*   **Libraries**: Progress Bars, Plotting, Console & GUI Apps, Scraping, and Web Apps.
*   **Multimedia**: NumPy, Image Manipulation, Animation, Audio, and Pygame.

## Table of Contents:

1.  [Collections](#1-collections) 🔗
2.  [Types](#2-types) 🔗
3.  [Syntax](#3-syntax) 🔗
4.  [System](#4-system) 🔗
5.  [Data](#5-data) 🔗
6.  [Advanced](#6-advanced) 🔗
7.  [Libraries](#7-libraries) 🔗
8.  [Multimedia](#8-multimedia) 🔗

## 1. Collections 🔗

*   **List**
    *   Creating, accessing, and modifying list elements.
    *   Sorting, reversing, and finding min/max/sum.
    *   Common list operations: `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `clear()`.
    *   List comprehensions and common operations like `zip()`.
*   **Dictionary**
    *   Creating dictionaries, accessing/modifying key-value pairs.
    *   Methods for keys, values, and items views (`.keys()`, `.values()`, `.items()`).
    *   `get()`, `setdefault()`, and `defaultdict()` for handling missing keys.
    *   Creating dictionaries from other collections.
    *   `update()`, `pop()`, and dictionary filtering.
    *   `collections.Counter`
*   **Set**
    *   Creating sets and set operations: `add()`, `update()`, `union()`, `intersection()`, `difference()`, `symmetric_difference()`, `issubset()`, `issuperset()`.
    *   Removing elements: `pop()`, `remove()`, `discard()`.
    *   `frozenset` (immutable set).
*   **Tuple**
    *   Creating tuples (immutable lists).
    *   Named tuples using `collections.namedtuple`.
*   **Range**
    *   Creating immutable sequences of integers.
*   **Enumerate**
    *   Iterating with index and element.
*   **Iterator**
    *   Understanding iterators and the `iter()` and `next()` functions.
    *   Itertools.
        *   `itertools.count`, `itertools.repeat`, `itertools.cycle`, `itertools.chain`, `itertools.islice`.
*   **Generator**
    *   Creating generators with `yield`.

## 2. Types 🔗

*   **Type**
    *   `type()` and `isinstance()`.
    *   Abstract Base Classes (ABCs): `Iterable`, `Collection`, `Sequence`, `Number`, `Complex`, `Real`, `Rational`, `Integral`.
*   **String**
    *   String methods: `strip()`, `split()`, `join()`, `startswith()`, `find()`.
    *   String methods: `lower()`, `upper()`, `replace()`, `translate()`.
    *   Character conversion: `chr()`, `ord()`.
    *   Property Methods: `isdecimal()`, `isdigit()`, `isnumeric()`, `isalnum()`, `isprintable()`, `isspace()`.
*   **Regex**
    *   Regular expression functions: `re.sub()`, `re.findall()`, `re.split()`, `re.search()`, `re.match()`, `re.finditer()`.
    *   Match object properties and special sequences.
*   **Format**
    *   f-strings and `.format()` for string formatting.
    *   Formatting options for strings, numbers, and floats.
*   **Numbers**
    *   Numeric types: `int`, `float`, `complex`, `fractions.Fraction`, `decimal.Decimal`.
    *   Built-in number functions: `pow()`, `abs()`, `round()`, `min()`, `sum()`.
    *   Math module: `floor()`, `ceil()`, `trunc()`, `pi`, `inf`, `nan`, `sqrt()`, `factorial()`, `sin()`, `cos()`, `tan()`, `log()`.
    *   Statistics module: `mean()`, `median()`, `mode()`, `variance()`, `stdev()`.
    *   Random module: `random()`, `randint()`, `uniform()`, `gauss()`, `choice()`, `shuffle()`.
    *   Hexadecimal numbers and bitwise operations.
*   **Combinatorics**
    *   `itertools.product()`, `itertools.permutations()`, `itertools.combinations()`.
*   **Datetime**
    *   `datetime` module: `date`, `time`, `datetime`, `timedelta`, `timezone`.
    *   Now, timezones, encoding/decoding, formatting, and arithmetic operations.

## 3. Syntax 🔗

*   **Function**
    *   Defining and calling functions.
    *   `Splat` operator and argument handling.
*   **Inline**
    *   Lambda functions.
    *   Comprehensions: list, generator, set, and dictionary comprehensions.
    *   `map()`, `filter()`, `reduce()`.
    *   `any()`, `all()`.
    *   Conditional expressions.
    *   `and` and `or` operators.
    *   Walrus operator.
    *   Named tuple, enum, and dataclass.
*   **Imports**
    *   Importing modules and packages.
*   **Closure**
    *   Understanding closures, and partial functions.
    *   Non-local variables.
*   **Decorator**
    *   Creating decorators and examples (debugger, cache, and parametrized decorators).
*   **Class**
    *   Creating classes.
    *   Special methods: `__init__()`, `__str__()`, `__repr__()`.
    *   Subclasses and inheritance.
    *   Type annotations.
    *   Dataclasses: `@dataclass` decorator.
    *   `@property`.
    *   Slots: `__slots__`.
    *   Copying objects.

## 4. System 🔗

*   **Exit**
    *   Exiting the interpreter: `sys.exit()`.
*   **Print**
    *   `print()` function and formatting.
    *   Pretty printing: `pprint()`.
*   **Input**
    *   Reading user input: `input()`.
*   **Command Line Arguments**
    *   Accessing command-line arguments: `sys.argv`.
    *   Using `argparse` to parse arguments.
*   **Open**
    *   Opening files: `open()`.
    *   File modes and exceptions.
    *   File object methods: `read()`, `readline()`, `readlines()`, `write()`, `writelines()`, `flush()`, `close()`.
*   **Paths**
    *   Working with paths: `os`, `glob`, and `pathlib`.
    *   `os.getcwd()`, `os.path.join()`, `os.path.basename()`, `os.path.dirname()`, `os.path.splitext()`.
    *   Listing files and directories: `os.listdir()`, `glob.glob()`, `os.scandir()`.
    *   Checking file/directory existence: `os.path.exists()`, `os.path.isfile()`, `os.path.isdir()`.
    *   File status: `os.stat()`.
    *   Path object methods.
*   **OS Commands**
    *   Operating system commands: `os.chdir()`, `os.mkdir()`, `os.makedirs()`, `shutil.copy()`, `shutil.copytree()`, `os.rename()`, `os.replace()`, `shutil.move()`, `os.remove()`, `os.rmdir()`, `shutil.rmtree()`.
    *   Shell commands: `os.popen()`, `subprocess.run()`.

## 5. Data 🔗

*   **JSON**
    *   Working with JSON: `json.dumps()`, `json.loads()`, `json.load()`, `json.dump()`.
*   **Pickle**
    *   Working with Pickle: `pickle.dumps()`, `pickle.loads()`, `pickle.load()`, `pickle.dump()`.
*   **CSV**
    *   Working with CSV files: `csv.reader()`, `csv.writer()`.
    *   CSV parameters and dialects.
    *   Reading and writing CSV files.
*   **SQLite**
    *   Connecting to SQLite databases: `sqlite3.connect()`.
    *   Executing queries and handling transactions.
    *   Using placeholders.
    *   SQLAlchemy.
*   **Bytes**
    *   Working with bytes: `bytes()`, `bytearray()`, `bytes.fromhex()`, `encode()`, `decode()`.
    *   Reading and writing bytes.
*   **Struct**
    *   Packing and unpacking data: `struct.pack()`, `struct.unpack()`.
*   **Array**
    *   Creating and using arrays: `array.array()`.
*   **Memory View**
    *   Creating and using memory views: `memoryview()`.
*   **Deque**
    *   Using double-ended queues: `collections.deque()`.

## 6. Advanced 🔗

*   **Operator**
    *   `operator` module for operator functions.
*   **Match Statement**
    *   Using the match statement.
    *   Patterns: value, class, wildcard, capture, as, or, sequence, mapping, class patterns.
*   **Logging**
    *   Setting up and using the `logging` module.
*   **Introspection**
    *   Inspecting objects: `dir()`, `vars()`, `hasattr()`, `getattr()`, `setattr()`, `delattr()`.
    *   Inspecting functions: `inspect.signature()`.
*   **Threading**
    *   Threading with `threading` and `concurrent.futures`.
    *   Threads, Locks, Semaphores, Events, Barriers, Queues, and Executors.
*   **Coroutines**
    *   Asynchronous programming with `asyncio`.
    *   Coroutines, tasks, gather, and wait.

## 7. Libraries 🔗

*   **Progress Bar**
    *   Using `tqdm`.
*   **Plot**
    *   Plotting with `matplotlib.pyplot`.
*   **Table**
    *   Printing tables with `tabulate`.
*   **Console App**
    *   Basic console application using `curses`.
*   **GUI App**
    *   GUI applications with `PySimpleGUI`.
*   **Scraping**
    *   Scraping with `requests`, `beautifulsoup4`, and `selenium`.
    *   XPath.
*   **Web App**
    *   Web applications with `flask`.

## 8. Multimedia 🔗

*   **NumPy**
    *   NumPy arrays and operations.
    *   Indexing, broadcasting, and common functions.
*   **Image**
    *   Image manipulation with `PIL.Image`.
    *   Image modes, pixel access, and drawing.
*   **Animation**
    *   Creating animations with `imageio`.
*   **Audio**
    *   Working with audio files using `wave` and `simpleaudio`.
    *   Reading/writing WAV files, and playing audio.
    *   Text to speech with `pyttsx3`.
*   **Synthesizer**
    *   Synthesizing sound.
*   **Pygame**
    *   Basic game development with `pygame`.
    *   Rect, Surface, drawing, sound, and example.
*   **Pandas**
    *   Series and DataFrames, merging, joining, and grouping.
    *   Reading and writing files with pandas.
*   **Plotly**
    *   Creating interactive plots with `plotly.express`.

## Appendix 🔗

*   **Cython**
    *   Introduction to Cython.
*   **Virtual Environments**
    *   Creating and using virtual environments.
*   **Basic Script Template**
    *   Script template for Python.
*   **Index**
    *   Index for easy navigation.
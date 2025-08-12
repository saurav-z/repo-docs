# Python Cheat Sheet: Your Comprehensive Guide to Python Programming 🔗

Python is a versatile and powerful programming language, perfect for beginners and experts alike. This cheat sheet provides a concise overview of Python's core concepts, from fundamental data structures to advanced libraries, helping you to quickly reference and master this dynamic language.  Access the complete reference and contribute at the original repository: [https://github.com/gto76/python-cheatsheet](https://github.com/gto76/python-cheatsheet).

## Key Features

*   **Comprehensive Coverage:** Dive deep into collections, types, syntax, system interactions, data handling, advanced techniques, libraries, and multimedia tools.
*   **Clear Syntax Examples:** Understand each concept with practical code snippets.
*   **Organized Structure:** Easily navigate the cheat sheet with a well-defined table of contents and section headings.
*   **Real-World Applications:** Explore libraries for data analysis, visualization, web development, and more.

## Table of Contents

1.  [Collections](#collections)
2.  [Types](#types)
3.  [Syntax](#syntax)
4.  [System](#system)
5.  [Data](#data)
6.  [Advanced](#advanced)
7.  [Libraries](#libraries)
8.  [Multimedia](#multimedia)

## Collections 🔗

*   **List** 🔗
    *   `list = [el_1, el_2, ...]` Creates a list object.
    *   `el = list[index]`  Access an element, starting index 0.
    *   `list.append(el)` Adds an element to the end.
    *   `list.sort()` Sorts elements in ascending order.
    *   `len(list)` Returns the number of items.
*   **Dictionary** 🔗
    *   `dict = {key_1: val_1, key_2: val_2, ...}` Create a dictionary object.
    *   `value = dict[key]` Access a value by its key.
    *   `dict.keys()` Returns a view of the keys.
    *   `dict.get(key, default=None)` Returns a value or a default value if the key is missing.
*   **Set** 🔗
    *   `set = {el_1, el_2, ...}` Creates a set.
    *   `set.add(el)` Adds an element.
    *   `set.union(coll)` Returns the union of sets.
    *   `set.intersection(coll)` Returns the intersection of sets.
*   **Tuple** 🔗
    *   Immutable list, use for fixed data.
    *   `tuple = (el_1, el_2, ...)` Create a tuple.
*   **Range** 🔗
    *   Immutable sequence of integers.
    *   `range(stop)` Generates a sequence from 0 up to *stop* (exclusive).
    *   `range(start, stop, step)` Generates a sequence with a specified step.
*   **Enumerate** 🔗
    *   Returns the index and value during iteration.
    *   `for i, el in enumerate(coll, start=0): ...`
*   **Iterator** 🔗
    *   Stream of elements.
    *   `iter(collection)` Creates an iterator from a collection.
    *   `next(iter, default)` Returns the next element or a default value if the iterator is exhausted.
*   **Generator** 🔗
    *   Function that yields values.
    *   `def count(start, step): yield start; start += step`

## Types 🔗

*   **Type** 🔗
    *   Everything is an object, and every object has a type.
    *   `type(el)` Returns the type of an object.
    *   `isinstance(el, type)` Checks if an object is of a specific type.
*   **String** 🔗
    *   Immutable sequence of characters.
    *   `str.strip()` Removes whitespace from both ends.
    *   `str.split()` Splits a string into a list of substrings.
    *   `str.join(coll_of_strings)` Joins elements into a string with the string as a separator.
*   **Regular_Exp** 🔗
    *   For pattern matching in strings.
    *   `re.search(r'<regex>', text)`  Searches for the first occurrence.
    *   `re.findall(r'<regex>', text)`  Finds all occurrences.
    *   `re.sub(r'<regex>', new, text)`  Substitutes all occurrences.
*   **Format** 🔗
    *   String formatting.
    *   `f'{el_1}, {el_2}'` Formatted string literals (f-strings).
    *   `'{0}, {a}'.format(el_1, a=el_2)`  String format method
*   **Numbers** 🔗
    *   Numeric data types.
    *   `int(value)` Converts to an integer.
    *   `float(value)` Converts to a floating-point number.
    *   `complex(real, imag)`  Creates a complex number.
*   **Combinatorics** 🔗
    *   Tools for permutations, combinations and product.
    *   `itertools.product()` Returns the Cartesian product of input iterables.
    *   `itertools.permutations()` Returns successive length permutations of elements.
    *   `itertools.combinations()` Returns r length subsequences of elements.
*   **Datetime** 🔗
    *   Working with dates and times.
    *   `datetime.date(year, month, day)` Create a date object.
    *   `datetime.datetime(year, month, day, hour, minute, second)` Create a datetime object.
    *   `datetime.timedelta(days, seconds, microseconds)` Represents a duration.

## Syntax 🔗

*   **Function** 🔗
    *   Reusable blocks of code.
    *   `def func_name(args): ... return value`
*   **Inline** 🔗
    *   Quick ways to write code.
    *   `lambda args: expression`  Anonymous function.
    *   `[expression for item in iterable if condition]`  List comprehension.
*   **Import** 🔗
    *   Including modules in your code.
    *   `import module`  Imports a module.
    *   `from module import object`  Imports specific objects from a module.
*   **Decorator** 🔗
    *   Adds functionality to functions.
    *   `@decorator_name def func(): ...`
*   **Class** 🔗
    *   Blueprint for creating objects.
    *   `class MyClass: ...`
    *   `__init__(self, ...)` Constructor.
*   **Duck_Type** 🔗
    *   Object's type is defined by its methods.
*   **Enum** 🔗
    *   Set of named constants.
    *   `from enum import Enum; class Color(Enum): RED = 1; GREEN = 2; BLUE = 3`
*   **Except** 🔗
    *   Handling errors.
    *   `try: ... except Exception as e: ... finally: ...`

## System 🔗

*   **Exit** 🔗
    *   Exiting the program.
    *   `sys.exit()`
*   **Print** 🔗
    *   Displaying output.
    *   `print(el_1, ..., sep=' ', end='\n', file=sys.stdout, flush=False)`
*   **Input** 🔗
    *   Getting user input.
    *   `input()`
*   **Command_Line_Arguments** 🔗
    *   Working with arguments passed to the script.
    *   `import sys; sys.argv`
    *   `argparse` module for parsing arguments.
*   **Open** 🔗
    *   Opening files.
    *   `open(path, mode='r', encoding=None)`
*   **Path** 🔗
    *   Working with file paths.
    *   `os.path`  module.
    *   `pathlib.Path` (modern approach).
*   **OS_Commands** 🔗
    *   Executing operating system commands.
    *   `os.chdir(path)`  Changing directories.
    *   `os.mkdir(path)`  Creating directories.
    *   `subprocess.run(...)` Running shell commands.

## Data 🔗

*   **JSON** 🔗
    *   Working with JSON data.
    *   `json.dumps(data)` Converts data to a JSON string.
    *   `json.loads(json_string)` Converts a JSON string to a Python object.
*   **Pickle** 🔗
    *   Serialization of Python objects.
    *   `pickle.dumps(object)` Converts to a bytes object.
    *   `pickle.loads(bytes)` Converts bytes to object.
*   **CSV** 🔗
    *   Reading and writing CSV files.
    *   `csv.reader(file)` and `csv.writer(file)`
*   **SQLite** 🔗
    *   Database operations.
    *   `sqlite3.connect(database_file)`
    *   `cursor.execute(query)`
*   **Bytes** 🔗
    *   Working with byte sequences.
    *   `bytes(string, 'utf-8')` Encode a string to bytes.
    *   `bytes.decode('utf-8')` Decode bytes to a string.
*   **Struct** 🔗
    *   Packing and unpacking binary data.
    *   `struct.pack(format, values)`
    *   `struct.unpack(format, bytes)`
*   **Array** 🔗
    *   Arrays of numeric data.
    *   `from array import array`
    *   `array('<typecode>', list_of_numbers)`
*   **Memory_View** 🔗
    *   Accessing memory.
    *   `memoryview(object)`
*   **Deque** 🔗
    *   Double-ended queue.
    *   `collections.deque`

## Advanced 🔗

*   **Operator** 🔗
    *   Operator functions for common operations.
    *   `import operator`
    *   `op.add(a, b)`, `op.mul(a, b)`, etc.
*   **Match_Stmt** 🔗
    *   Pattern matching.
    *   `match variable: case pattern: ...`
*   **Logging** 🔗
    *   Logging messages.
    *   `import logging`
    *   `logging.debug/info/warning/error/critical(...)`
*   **Introspection** 🔗
    *   Examining objects.
    *   `dir(obj)`, `vars(obj)`, `hasattr(obj, name)`
*   **Threading** 🔗
    *   Multithreading.
    *   `import threading`
    *   `Thread(target=function, args=arguments)`
*   **Coroutines** 🔗
    *   Asynchronous programming.
    *   `import asyncio`
    *   `async def func(): ... await coroutine()`

## Libraries 🔗

*   **Progress_Bar** 🔗
    *   Displaying progress.
    *   `from tqdm import tqdm`
*   **Plot** 🔗
    *   Plotting charts and graphs.
    *   `import matplotlib.pyplot as plt`
    *   `plt.plot(x, y)` and other plotting functions.
*   **Table** 🔗
    *   Displaying tabular data.
    *   `from tabulate import tabulate`
*   **Console_App** 🔗
    *   Building console applications.
    *   `import curses`
*   **GUI** 🔗
    *   Creating graphical user interfaces.
    *   `import PySimpleGUI as sg`
*   **Scraping** 🔗
    *   Web scraping.
    *   `import requests, bs4`
    *   `from selenium import webdriver` (for dynamic content).
*   **Web** 🔗
    *   Web frameworks.
    *   `import flask as fl` (microframework)
*   **Profile** 🔗
    *   Profiling and performance analysis.
    *   `import timeit`
    *   `import cProfile`

## Multimedia 🔗

*   **NumPy** 🔗
    *   Numerical computing.
    *   `import numpy as np`
*   **Image** 🔗
    *   Image processing.
    *   `from PIL import Image`
*   **Animation** 🔗
    *   Creating animations.
    *   `import imageio`
*   **Audio** 🔗
    *   Working with audio.
    *   `import wave`
    *   `import simpleaudio`
*   **Synthesizer** 🔗
    *   Creating audio synthesis.
*   **Pygame** 🔗
    *   Game development.
    *   `import pygame as pg`
*   **Pandas** 🔗
    *   Data analysis and manipulation.
    *   `import pandas as pd`
*   **Plotly** 🔗
    *   Interactive plotting and data visualization.
    *   `import plotly.express as px`
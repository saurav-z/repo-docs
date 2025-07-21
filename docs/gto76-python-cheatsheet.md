# Comprehensive Python Cheatsheet 🔗

**Level up your Python skills with this comprehensive cheatsheet, covering everything from fundamental syntax to advanced libraries.**  This resource is your one-stop guide for mastering Python, from basic concepts to advanced techniques.  [Find the original repository here](https://github.com/gto76/python-cheatsheet).

## Key Features:

*   **Comprehensive Coverage:**  From collections and types to advanced concepts like multithreading and web applications.
*   **Practical Examples:**  Real-world code snippets to illustrate each concept.
*   **Organized Structure:**  Clear headings, subheadings, and lists for easy navigation.
*   **SEO Optimized:**  Keywords and structure designed for easy searching and quick reference.

## Table of Contents:

1.  [Collections](#collections)
2.  [Types](#types)
3.  [Syntax](#syntax)
4.  [System](#system)
5.  [Data](#data)
6.  [Advanced](#advanced)
7.  [Libraries](#libraries)
8.  [Multimedia](#multimedia)

---

## 1. Collections 🔗

### List 🔗

*   **Creation:** `list = [<el_1>, <el_2>, ...]` or `list(<collection>)`
*   **Access:** `<el> = list[index]` or `<list> = list[from_inclusive : to_exclusive : ±step]`
*   **Manipulation:** `append()`, `extend()`, `sort()`, `reverse()`, `sorted()`, `reversed()`, `max()`, `sum()`, `len()`, `count()`, `index()`, `pop()`, `insert()`, `remove()`, `clear()`

### Dictionary 🔗

*   **Creation:** `dict = {key_1: val_1, key_2: val_2, ...}`
*   **Access:** `value = dict[key]`, `keys()`, `values()`, `items()`
*   **Special:** `get()`, `setdefault()`, `defaultdict()`, `dict.fromkeys()`, `update()`, `pop()`

### Counter 🔗

*   **Usage:** `from collections import Counter` for counting item occurrences.

### Set 🔗

*   **Creation:** `set = {<el_1>, <el_2>, ...}`
*   **Manipulation:** `add()`, `update()`, `union()`, `intersection()`, `difference()`, `symmetric_difference()`, `issubset()`, `issuperset()`, `pop()`, `remove()`, `discard()`

### Frozen Set 🔗

*   **Creation:** `frozenset(<collection>)` (immutable, hashable set)

### Tuple 🔗

*   **Creation:** `tuple = ()` or `(<el>,)` or `(<el_1>, <el_2> [, ...])` (immutable list)

### Named Tuple 🔗

*   **Creation:** `from collections import namedtuple`, `Point = namedtuple('Point', 'x y')`

### Range 🔗

*   **Creation:** `range(stop)`, `range(start, stop)`, `range(start, stop, ±step)` (immutable sequence of integers)

### Enumerate 🔗

*   **Usage:** `for i, el in enumerate(collection, start=0): ...` (returns index and element)

### Iterator 🔗

*   **Creation:** `iter(<collection>)`, `iter(<function>, to_exclusive)`
*   **Usage:** `next(<iter> [, default])`, `list(<iter>)`

### Itertools 🔗

*   **Usage:** `import itertools as it`
    *   `count()`, `repeat()`, `cycle()`, `chain()`, `chain.from_iterable()`, `islice()`

### Generator 🔗

*   **Creation:** Functions with `yield` statements (interchangeable with iterators)

---

## 2. Types 🔗

### Type 🔗

*   **Usage:** `type(<el>)` or `<el>.__class__`, `isinstance(<el>, <type>)`

### Abstract Base Classes 🔗

*   **Usage:** `from collections.abc import Iterable, Collection, Sequence`
    *   `from numbers import Number, Complex, Real, Rational, Integral` (for type checking)

### String 🔗

*   **Manipulation:** `strip()`, `split()`, `splitlines()`, `join()`, `lower()`, `upper()`, `casefold()`, `replace()`, `translate()`, `chr()`, `ord()`
*   **Properties:** `isdecimal()`, `isdigit()`, `isnumeric()`, `isalnum()`, `isprintable()`, `isspace()`

### Regex 🔗

*   **Usage:** `import re`
    *   `sub()`, `findall()`, `split()`, `search()`, `match()`, `finditer()`
*   **Match Object:** `group()`, `groups()`, `start()`, `end()`
*   **Special Sequences:** `\d`, `\w`, `\s`
*   **Flags:**  `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL`, `re.ASCII`

### Format 🔗

*   **Usage:** `f'{<el_1>}, {<el_2>}'`, `'{}, {}'.format(<el_1>, <el_2>)`, `'%s, %s' % (<el_1>, <el_2>)`
*   **Options:** Alignment, precision, number formatting (strings, numbers, floats, ints)

### Numbers 🔗

*   **Types:** `int`, `float`, `complex`, `fractions.Fraction`, `decimal.Decimal`
*   **Built-in Functions:** `pow()`, `abs()`, `round()`, `min()`, `sum()`
*   **Math:** `math.floor`, `math.ceil`, `math.trunc`, `math.pi`, `math.inf`, `math.isnan`, `math.sqrt`, `math.factorial`, `math.sin`, `math.cos`, `math.tan`, `math.log`, `math.log10`, `math.log2`
*   **Statistics:** `statistics.mean`, `statistics.median`, `statistics.mode`, `statistics.variance`, `statistics.stdev`
*   **Random:** `random.random`, `random.randint`, `random.uniform`, `random.gauss`, `random.choice`, `random.shuffle`, `random.seed`
*   **Hexadecimal:** `0x<hex>`, `int('<hex>', 16)`, `hex(<int>)`, `bin()`
*   **Bitwise Operators:** `&`, `|`, `^`, `<<`, `>>`, `~`

### Combinatorics 🔗

*   **Usage:** `import itertools as it`
    *   `product()`, `permutations()`, `combinations()`

### Datetime 🔗

*   **Usage:** `from datetime import date, time, datetime, timedelta, timezone`
    *   `import zoneinfo, dateutil.tz`
*   **Classes:** `date`, `time`, `datetime`, `timedelta`
*   **Now:**  `today()`, `now()`, `astimezone()`
*   **Timezone:** `timezone.utc`, `timezone(<timedelta>)`, `dateutil.tz.tzlocal()`, `zoneinfo.ZoneInfo('<iana_key>')`,
*   **Encode/Decode:** `fromisoformat()`, `strptime()`, `fromordinal()`, `fromtimestamp()`, `isoformat()`, `strftime()`, `toordinal()`, `timestamp()`
*   **Format:** `strptime()`, `strftime()` (format codes)
*   **Arithmetics:** `+`, `-`, `*`, `/`, `±`
*   **Time delta**

---

## 3. Syntax 🔗

### Function 🔗

*   **Definition:** `def <func_name>(<nondefault_args>, <default_args>): ...`
*   **Calling:** `<obj> = <function>(<positional_args>, <keyword_args>)`
*   **Splat Operator:**
    *   `*args` (packs positional arguments)
    *   `**kwargs` (packs keyword arguments)

### Inline 🔗

*   **Lambda:** `lambda <arg_1>, <arg_2>: <return_value>` (single-expression functions)
*   **Comprehensions:** `[<exp> for <item> in <coll> if <condition>]` (list, set, dictionary)
*   **Map, Filter, Reduce:** `from functools import reduce`
    *   `map()`, `filter()`, `reduce()`
*   **Any, All:** `any(<collection>)`, `all(<collection>)`
*   **Conditional Expression:** `<exp> if <condition> else <exp>`
*   **And, Or:** `<exp> and <exp>`, `<exp> or <exp>`
*   **Walrus Operator:** `[i for a in '0123' if (i := int(a)) > 0]`
*   **Named Tuple, Enum, Dataclass:** (create subclasses)

### Imports 🔗

*   **Usage:** `import <module>`, `import <package>.<module>`, `from <module> import <obj>`

### Closure 🔗

*   **Example:** Nested function referencing an outer function's variable.

### Partial 🔗

*   **Usage:** `from functools import partial`, `partial(<function>, <arg_1>, ...)` (partially apply function arguments)

### Non-Local 🔗

*   **Usage:** `nonlocal i` (access variables in enclosing scopes)

### Decorator 🔗

*   **Definition:** `@decorator_name` (function that modifies another function)

### Class 🔗

*   **Definition:** `class MyClass:`
    *   `__init__(self, a):` (constructor)
    *   `__str__(self):` (string representation)
    *   `__repr__(self):` (unambiguous representation)
    *   `@classmethod`, `@staticmethod`
*   **Subclassing:** `class Employee(Person):` (inheritance)
    *   `super().__init__(name)`
*   **Type Annotations:** `name: str = "John"`
*   **Dataclass:**  `from dataclasses import dataclass`
    *   `@dataclass` (automatically generates methods)
*   **Property:** `@property`, `@name.setter` (getters and setters)
*   **Slots:** `__slots__ = ['a']` (memory optimization)
*   **Copy:** `from copy import copy, deepcopy`

### Duck Types 🔗

*   **Comparable:** `__eq__(self, other)`, `__ne__(self, other)`, `__lt__(self, other)`, `__gt__(self, other)`, `__le__(self, other)`, `__ge__(self, other)`
*   **Hashable:** `__hash__(self)`
*   **Sortable:**  `from functools import total_ordering`
    *   `@total_ordering` (simplifies sortable classes)
*   **Iterator:**  `__next__(self)`, `__iter__(self)`
*   **Callable:** `__call__(self)`
*   **Context Manager:**  `__enter__(self)`, `__exit__(self, exc_type, exception, traceback)`
*   **Iterable, Collection, Sequence**: `iter()`, `len()`, `getitem()`

### Enum 🔗

*   **Usage:** `from enum import Enum, auto`
    *   `class <enum_name>(Enum):`
    *   `<member_name> = auto()`, `<member_name> = <value>`

### Exceptions 🔗

*   **Usage:** `try...except...else...finally`
    *   `except <exception> as <name>:`
    *   `raise <exception>`
    *   **Built-in Exceptions:** `Exception` and its subclasses
    *   **User-defined Exceptions:** `class MyError(Exception): pass`

---

## 4. System 🔗

### Exit 🔗

*   **Usage:** `import sys`, `sys.exit()`, `sys.exit(<int>)`

### Print 🔗

*   **Usage:** `print(<el_1>, ..., sep=' ', end='\n', file=sys.stdout, flush=False)`
*   **Pretty Print:** `from pprint import pprint`

### Input 🔗

*   **Usage:** `<str> = input()`

### Command Line Arguments 🔗

*   **Usage:** `import sys`, `sys.argv`
*   **Argument Parser:** `from argparse import ArgumentParser`
    *   `add_argument()`, `parse_args()`

### Open 🔗

*   **Usage:** `open(<path>, mode='r', encoding=None, newline=None)`
    *   **Modes:** `r`, `w`, `x`, `a`, `w+`, `r+`, `a+`, `b`
    *   **Exceptions:** `FileNotFoundError`, `FileExistsError`, `IsADirectoryError`, `PermissionError`, `OSError`
*   **File Object:** `read()`, `readline()`, `readlines()`, `write()`, `writelines()`, `flush()`, `close()`, `seek()`

### Paths 🔗

*   **Usage:** `import os, glob`, `from pathlib import Path`
    *   **Paths:** `os.getcwd()`, `os.path.join()`, `os.path.realpath()`
    *   `os.path.basename()`, `os.path.dirname()`, `os.path.splitext()`
    *   `os.listdir()`, `glob.glob()`
    *   `os.path.exists()`, `os.path.isfile()`, `os.path.isdir()`
    *   `os.stat()`
    *   `os.scandir()`, `DirEntry`
    *   `Path()`, `Path.cwd()`, `Path.home()`, `/`, `.resolve()`, `.parent`, `.name`, `.suffix`, `.stem`, `.parts`, `.iterdir()`, `.glob()`, `open()`

### OS Commands 🔗

*   **Usage:** `import os, shutil, subprocess`
    *   `os.chdir()`, `os.mkdir()`, `os.makedirs()`
    *   `shutil.copy()`, `shutil.copy2()`, `shutil.copytree()`
    *   `os.rename()`, `os.replace()`, `shutil.move()`
    *   `os.remove()`, `os.rmdir()`, `shutil.rmtree()`
    *   `os.popen()`, `subprocess.run()`

---

## 5. Data 🔗

### JSON 🔗

*   **Usage:** `import json`
    *   `dumps()`, `loads()`

### Pickle 🔗

*   **Usage:** `import pickle`
    *   `dumps()`, `loads()`

### CSV 🔗

*   **Usage:** `import csv`
    *   `reader()`, `next()`, `writer()`, `writerow()`, `writerows()`
    *   **Dialects:** `excel`, `excel-tab`, `unix`

### SQLite 🔗

*   **Usage:** `import sqlite3`
    *   `connect()`, `close()`, `execute()`, `fetchone()`, `fetchall()`, `commit()`, `rollback()`
*   **SQLAlchemy:** `from sqlalchemy import create_engine, text`
    *   `create_engine()`, `connect()`, `execute()`

### Bytes 🔗

*   **Creation:** `b'<str>'`,  `bytes(<coll_of_ints>)`, `bytes(<str>, 'utf-8')`, `bytes.fromhex('<hex>')`, `<int>.to_bytes(n_bytes, …)`
*   **Encode/Decode:**  `encode()`, `decode()`, `fromhex()`, `to_bytes()`

### Struct 🔗

*   **Usage:** `from struct import pack, unpack`
    *   `pack()`, `unpack()`
    *   **Format Codes:**  `=`, `<`, `>`, `c`, `<n>s`, `b`, `h`, `i`, `l`, `q`, `f`, `d`

### Array 🔗

*   **Usage:** `from array import array`
    *   `array('<typecode>', <coll_of_nums>)`
    *   `fromfile()`, `bytes()`, `write()`

### Memory View 🔗

*   **Usage:** `memoryview(<bytes/bytearray/array>)`
    *   `cast('<typecode>')`, `release()`

### Deque 🔗

*   **Usage:** `from collections import deque`
    *   `appendleft()`, `extendleft()`, `rotate()`, `popleft()`

---

## 6. Advanced 🔗

### Operator 🔗

*   **Usage:** `import operator as op`
    *   Operator functions for common operations (comparison, arithmetic, etc.)

### Match Statement 🔗

*   **Usage:** `match <object/expression>:`
    *   `case <pattern> [if <condition>]:`

### Patterns 🔗

*   `<value_pattern>`, `<class_pattern>`, `_`, `<capture_patt>`, `<as_pattern>`, `<or_pattern>`, `<sequence_patt>`, `<mapping_patt>`, `<class_pattern>`

### Logging 🔗

*   **Usage:** `import logging as log`
    *   `basicConfig()`, `debug()`, `info()`, `warning()`, `error()`, `critical()`, `getLogger()`, `exception()`
    *   `Formatter`, `FileHandler`

### Introspection 🔗

*   **Usage:** `dir()`, `vars()`, `globals()`
    *   `dir(<obj>)`, `vars(<obj>)`, `hasattr()`, `getattr()`, `setattr()`, `delattr()`
    *   `inspect.signature()`, `Parameter`

### Threading 🔗

*   **Usage:** `from threading import Thread, Lock, RLock, Semaphore, Event, Barrier`
    *   `from concurrent.futures import ThreadPoolExecutor, as_completed`
    *   `Thread.start()`, `Thread.join()`
    *   `Lock.acquire()`, `Lock.release()`
    *   `Semaphore()`, `Event()`, `Barrier()`
    *   `Queue.put()`, `Queue.get()`, `Queue.put_nowait()`, `Queue.get_nowait()`
    *   `ThreadPoolExecutor.map()`, `ThreadPoolExecutor.submit()`, `ThreadPoolExecutor.shutdown()`, `as_completed()`

### Coroutines 🔗

*   **Usage:** `import asyncio as aio`
    *   `async def`, `await`, `aio.create_task()`, `aio.gather()`, `aio.wait()`, `aio.as_completed()`

---

## 7. Libraries 🔗

### Progress Bar 🔗

*   **Usage:** `import tqdm`, `for el in tqdm.tqdm([1, 2, 3], desc='Processing'): ...`

### Plot 🔗

*   **Usage:** `import matplotlib.pyplot as plt`
    *   `plot()`, `bar()`, `scatter()`, `legend()`, `title()`, `xlabel()`, `ylabel()`, `show()`, `savefig()`, `clf()`

### Table 🔗

*   **Usage:** `import csv, tabulate`
    *   `tabulate.tabulate(rows, headers='firstrow')`

### Console App 🔗

*   **Example:** Basic file explorer using `curses`

### GUI App 🔗

*   **Example:** Weight converter using `PySimpleGUI`

### Scraping 🔗

*   **Usage:** `import requests, bs4` (Beautiful Soup)
    *   `requests.get()`, `bs4.BeautifulSoup()`
    *   **Selenium:** `from selenium import webdriver`
        *   `webdriver.Chrome()`, `get()`, `page_source`, `find_element()`, `find_elements()`, `get_attribute()`, `click()`, `clear()`, `send_keys()`

### Web App 🔗

*   **Usage:** `import flask as fl`
    *   `app = fl.Flask(__name__)`
    *   `app.route()`, `render_template_string()`, `send_from_directory()`, `abort()`, `redirect()`, `request.args`, `session`

### Profiling 🔗

*   **Timing:** `from timeit import timeit`
*   **Line Profiling:** `pip3 install line_profiler`
    *   `kernprof -lv test.py`
*   **Call and Flame Graphs:**
    *   `pip3 install gprof2dot snakeviz`
    *   `python3 -m cProfile -o test.prof test.py`
    *   `gprof2dot --format=pstats test.prof | dot -T png -o test.png`
    *   `snakeviz test.prof`
*   **Sampling and Memory Profilers:** (various libraries for CPU and memory profiling)

---

## 8. Multimedia 🔗

### NumPy 🔗

*   **Usage:** `import numpy as np`
    *   `array()`, `zeros()`, `ones()`, `empty()`, `arange()`, `linspace()`, `random.randint()`, `random.random()`, `reshape()`, `flatten()`, `ravel()`, `transpose()`, `copy()`, `abs()`, `sqrt()`, `log()`, `int64()`, `sum()`, `max()`, `mean()`, `argmax()`, `all()`, `apply_along_axis()`, `concatenate()`, `vstack()`, `column_stack()`, `tile()`, `repeat()`
    *   **Indexing:** (advanced indexing and broadcasting)

### Image 🔗

*   **Usage:** `from PIL import Image`
    *   `Image.new()`, `Image.open()`, `Image.convert()`, `Image.save()`, `Image.show()`, `getpixel()`, `getdata()`, `putpixel()`, `putdata()`, `paste()`, `filter()`, `enhance()`
    *   **Modes:** `L`, `RGB`, `RGBA`, `HSV`
    *   **Image Draw:** `from PIL import ImageDraw`
        *   `Draw.Draw()`, `point()`, `line()`, `arc()`, `rectangle()`, `polygon()`, `ellipse()`, `text()`

### Animation 🔗

*   **Example:** GIF creation using `imageio`

### Audio 🔗

*   **Usage:** `import wave`
    *   `wave.open()`, `getframerate()`, `getnchannels()`, `getsampwidth()`, `getparams()`, `readframes()`, `setframerate()`, `setnchannels()`, `setsampwidth()`, `setparams()`, `writeframes()`

### Synthesizer 🔗

*   **Example:** Music generation using `simpleaudio`

### Pygame 🔗

*   **Usage:** `import pygame as pg`
    *   `pg.init()`, `set_mode()`, `Rect`, `Surface`, `image.load()`, `Surface.fill()`, `Surface.set_at()`, `Surface.blit()`, `pg.draw.rect`, `pg.display.flip()`, `pg.quit()`
    *   **Basic Example:**  Mario Brothers

### Pandas 🔗

*   **Usage:** `import pandas as pd`
    *   Series and DataFrame creation, indexing, manipulation, plotting, merging, grouping, etc.

### Plotly 🔗

*   **Usage:** `import plotly.express as px, pandas as pd`
    *   `px.line()`, `px.area()`, `px.bar()`, `px.scatter()`, `px.scatter_3d()`, `px.histogram()`

---

## Appendix 🔗

### Cython 🔗

*   **Usage:**  `import pyximport; pyximport.install()`, `import <cython_script>`

### Virtual Environments 🔗

*   **Usage:** `python3 -m venv NAME`, `source NAME/bin/activate`, `pip3 install LIBRARY`, `python3 FILE`, `deactivate`

### Basic Script Template 🔗

*   **Example:**  Template for a basic Python script
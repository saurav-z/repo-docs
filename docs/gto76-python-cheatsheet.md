# Comprehensive Python Cheatsheet: Your Go-To Guide 🔗

**Unlock the power of Python with this comprehensive cheatsheet, packed with essential concepts, syntax, and practical examples to boost your coding efficiency.**

- **Collections:** Lists, Dictionaries, Sets, Tuples, and more.
- **Types:** Strings, Numbers, Regular Expressions, and Date/Time.
- **Syntax:** Functions, Classes, Imports, Decorators, and more.
- **System:** File I/O, Command-Line Arguments, and OS Commands.
- **Data:** JSON, Pickle, CSV, and SQLite.
- **Advanced:** Operators, Logging, and Concurrency.
- **Libraries:** Plotting, GUI, Scraping, and Web Apps.
- **Multimedia:** NumPy, Image Manipulation, Audio, and Pygame.

***
[Download text file](https://raw.githubusercontent.com/gto76/python-cheatsheet/main/README.md), [Fork me on GitHub](https://github.com/gto76/python-cheatsheet)

---
## **1. Collections: 🔗**

*   **Lists: 🔗**  Ordered, mutable sequences.
    ```python
    <list> = [<el_1>, <el_2>, ...]  # Creates a list
    <el>   = <list>[index]          # Access by index (0-based)
    <list>.append(<el>)             # Add element to the end
    <list>.sort()                   # Sort elements
    <int>  = len(<list>)             # Get the number of items
    ```

*   **Dictionaries: 🔗** Key-value pairs, unordered.
    ```python
    <dict> = {key_1: val_1, key_2: val_2, ...}  # Create a dictionary
    value  = <dict>.get(key, default=None)     # Get value safely
    <dict>.update(<dict>)                       # Add or update items
    ```

*   **Sets: 🔗** Unordered collections of unique elements.
    ```python
    <set> = {<el_1>, <el_2>, ...}  # Create a set
    <set>.add(<el>)                 # Add an element
    <set> = <set>.union(<coll.>)    # Combine sets
    ```

*   **Tuples: 🔗** Ordered, *immutable* sequences.
    ```python
    <tuple> = (<el_1>, <el_2>, ...)  # Create a tuple
    ```

*   **Ranges: 🔗** Immutable sequences of integers.
    ```python
    <range> = range(stop)          # range(to_exclusive)
    <range> = range(start, stop)   # range(from_inclusive, to_exclusive)
    ```

*   **Enumerate: 🔗**  Iterate with indices.
    ```python
    for i, el in enumerate(<coll>, start=0):
        ...
    ```

*   **Iterators: 🔗**  Generate a sequence of values.
    ```python
    <iter> = iter(<collection>)
    <el>   = next(<iter> [, default])
    ```

*   **Generators: 🔗** Functions that yield values.
    ```python
    def my_generator():
        yield value
    ```

---

## **2. Types: 🔗**

*   **Type: 🔗**  Everything is an object, every object has a type.
    ```python
    <type> = type(<el>)
    <bool> = isinstance(<el>, <type>)
    ```

*   **String: 🔗** Immutable sequence of characters.
    ```python
    <str>  = <str>.strip()                    # Remove whitespace
    <list> = <str>.split()                    # Split into substrings
    <str>  = <str>.join(<coll_of_strings>)    # Join strings
    <bool> = <sub_str> in <str>               # Check for substring
    <str>  = <str>.lower()                    # Convert to lowercase
    ```

*   **Regex: 🔗**  Regular expression matching.
    ```python
    import re
    <list>  = re.findall(r'<regex>', text)    # Find all matches
    <Match> = re.search(r'<regex>', text)     # Find the first match
    <str>   = re.sub(r'<regex>', new, text)   # Replace matches
    ```

*   **Format: 🔗** String formatting.
    ```python
    <str> = f'{<el_1>}, {<el_2>}'
    <str> = '{}, {}'.format(<el_1>, <el_2>)
    ```

*   **Numbers: 🔗** Integers, floats, complex numbers.
    ```python
    <int>      = int(<float/str/bool>)
    <float>    = float(<int/str/bool>)
    <complex>  = complex(real=0, imag=0)
    ```

*   **Combinatorics: 🔗**  Generate permutations, combinations.
    ```python
    import itertools as it
    <iter> = it.product('abc', repeat=2)     # Cartesian product
    <iter> = it.permutations('abc', 2)       # Permutations
    <iter> = it.combinations('abc', 2)       # Combinations
    ```

*   **Datetime: 🔗** Date and time manipulation.
    ```python
    from datetime import datetime
    <DT> = datetime(year, month, day, hour=0) # Create datetime
    <str> = <DT>.strftime('%Y-%m-%d %H:%M:%S') # Format datetime to string
    ```

---

## **3. Syntax: 🔗**

*   **Function: 🔗** Reusable blocks of code.
    ```python
    def <func_name>(<nondefault_args>, <default_args>):
        ...
    ```

*   **Inline: 🔗**  Lambda functions, comprehensions, map/filter.
    ```python
    <func> = lambda <arg>: <return_value>    # Lambda function
    <list> = [i+1 for i in range(10)]       # List comprehension
    <iter> = map(lambda x: x + 1, range(10)) # Map
    ```

*   **Imports: 🔗**  Bring in external code.
    ```python
    import <module>
    from <module> import <object>
    ```

*   **Decorator: 🔗** Modify function behavior.
    ```python
    @my_decorator
    def my_function():
        ...
    ```

*   **Class: 🔗** Create custom objects.
    ```python
    class MyClass:
        def __init__(self, a):
            self.a = a
        def __str__(self):
            return str(self.a)
    ```

*   **Duck Types: 🔗** Based on what methods an object supports.
    ```python
    #Example: A Comparable type would implement __eq__
    class MyComparable:
        def __init__(self, a):
            self.a = a
        def __eq__(self, other):
            if isinstance(other, type(self)):
                return self.a == other.a
            return NotImplemented
    ```

*   **Enum: 🔗**  Create named constants.
    ```python
    from enum import Enum
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    ```

*   **Except: 🔗**  Handle errors.
    ```python
    try:
        # code that might raise an exception
    except <exception>:
        # code to handle the exception
    ```

---

## **4. System: 🔗**

*   **Exit: 🔗** Terminate the program.
    ```python
    import sys
    sys.exit()
    ```

*   **Print: 🔗**  Display output.
    ```python
    print(<el_1>, ..., sep=' ', end='\n', file=sys.stdout)
    ```

*   **Input: 🔗** Get user input.
    ```python
    <str> = input()
    ```

*   **Command Line Arguments: 🔗**  Access arguments.
    ```python
    import sys
    scripts_path = sys.argv[0]
    arguments    = sys.argv[1:]
    ```
    ```python
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-<short_name>', '--<name>', type=<type>, help='example')
    args  = p.parse_args()
    ```

*   **Open: 🔗**  Work with files.
    ```python
    <file> = open(<path>, mode='r', encoding='utf-8')
    ```

*   **Path: 🔗**  Handle file paths.
    ```python
    import os
    <str>  = os.getcwd()                 # Get current directory
    <bool> = os.path.exists(<path>)     # Check if path exists
    ```
    ```python
    from pathlib import Path
    <Path> = Path('<path>')
    ```

*   **OS Commands: 🔗** Execute OS-level commands.
    ```python
    import os
    os.mkdir(<path>)
    import shutil
    shutil.copy(from_path, to_path)
    ```

---

## **5. Data: 🔗**

*   **JSON: 🔗**  Work with JSON data.
    ```python
    import json
    <str>  = json.dumps(<list/dict>)   # Serialize to JSON
    <coll> = json.loads(<str>)         # Deserialize from JSON
    ```

*   **Pickle: 🔗**  Serialize and deserialize Python objects.
    ```python
    import pickle
    <bytes>  = pickle.dumps(<object>)  # Serialize to bytes
    <object> = pickle.loads(<bytes>)   # Deserialize from bytes
    ```

*   **CSV: 🔗**  Work with CSV files.
    ```python
    import csv
    <reader> = csv.reader(<file>)    # Read CSV
    <writer> = csv.writer(<file>)    # Write CSV
    ```

*   **SQLite: 🔗**  Interact with SQLite databases.
    ```python
    import sqlite3
    <conn> = sqlite3.connect('<database>.db')
    <cursor> = <conn>.execute('<query>')
    <conn>.commit()
    ```

*   **Bytes: 🔗**  Work with byte data.
    ```python
    <bytes> = b'<str>'               # Create bytes object
    <bytes> = <str>.encode('utf-8')    # Encode string to bytes
    <str>   = <bytes>.decode('utf-8')    # Decode bytes to string
    ```

*   **Struct: 🔗**  Pack and unpack binary data.
    ```python
    from struct import pack, unpack
    <bytes> = pack('<format>', <el_1>, ...)
    <tuple> = unpack('<format>', <bytes>)
    ```

*   **Array: 🔗**  Arrays of numeric data.
    ```python
    from array import array
    <array> = array('<typecode>', <coll_of_nums>)
    ```

*   **Memory View: 🔗** Access internal data of an object without copying.
    ```python
    <mview> = memoryview(<bytes/bytearray/array>)
    ```

*   **Deque: 🔗** Double-ended queue.
    ```python
    from collections import deque
    <deque> = deque(<collection>)
    <deque>.appendleft(<el>)
    <el> = <deque>.popleft()
    ```

---

## **6. Advanced: 🔗**

*   **Operator: 🔗**  Operator functions.
    ```python
    import operator as op
    <bool> = op.eq/ne/lt/ge/is_/is_not/contains(<obj>, <obj>)
    <num>  = op.add/sub/mul/truediv/floordiv/mod(<obj>, <obj>)
    ```

*   **Match Statement: 🔗** Pattern matching.
    ```python
    match <object>:
        case <pattern>:
            <code>
        ...
    ```

*   **Logging: 🔗**  Log messages.
    ```python
    import logging
    logging.basicConfig(level='DEBUG')
    logging.debug('<message>')
    ```

*   **Introspection: 🔗**  Inspect objects.
    ```python
    import inspect
    <dict> = vars(<obj>)  # Get object attributes
    ```

*   **Threading: 🔗**  Concurrent execution.
    ```python
    from threading import Thread, Lock
    <Thread> = Thread(target=<function>)
    <Thread>.start()
    <lock> = Lock()
    with <lock>:
        # Access a shared resource
    ```

*   **Coroutines: 🔗**  Asynchronous programming.
    ```python
    import asyncio
    async def my_coroutine():
        await asyncio.sleep(1)  # non-blocking
    asyncio.run(my_coroutine())
    ```

---

## **7. Libraries: 🔗**

*   **Progress Bar: 🔗** Display progress.
    ```python
    # $ pip3 install tqdm
    import tqdm
    for el in tqdm.tqdm(<iterable>, desc='Description'):
       ...
    ```

*   **Plot: 🔗** Generate plots.
    ```python
    # $ pip3 install matplotlib
    import matplotlib.pyplot as plt
    plt.plot/bar/scatter(x_data, y_data)
    plt.show()
    ```

*   **Table: 🔗**  Display data in tables.
    ```python
    # $ pip3 install tabulate
    import tabulate, csv
    with open('data.csv') as f:
      rows = list(csv.reader(f))
    print(tabulate.tabulate(rows, headers='firstrow'))
    ```

*   **Console App: 🔗** Create console-based applications.
    ```python
    # Example: File Explorer (see original repo)
    ```

*   **GUI App: 🔗**  Build graphical user interfaces.
    ```python
    # $ pip3 install PySimpleGUI
    import PySimpleGUI as sg
    window = sg.Window(...)
    event, values = window.read()
    ```

*   **Scraping: 🔗**  Extract data from websites.
    ```python
    # $ pip3 install requests beautifulsoup4
    import requests, bs4
    response = requests.get('<url>')
    document = bs4.BeautifulSoup(response.text, 'html.parser')
    element = document.find(...)
    ```

*   **Web App: 🔗**  Build web applications.
    ```python
    # $ pip3 install flask
    import flask as fl
    app = fl.Flask(__name__)
    @app.route('/')
    def index():
       return 'Hello, World!'
    app.run()
    ```

*   **Profiling: 🔗** Analyze performance.
    ```python
    # See detailed methods in Profiling Section above
    ```

---

## **8. Multimedia: 🔗**

*   **NumPy: 🔗**  Numerical computing with arrays.
    ```python
    # $ pip3 install numpy
    import numpy as np
    <array> = np.array(<list>)
    <array> = <array>.reshape(<shape>)
    ```

*   **Image: 🔗**  Image manipulation.
    ```python
    # $ pip3 install pillow
    from PIL import Image
    <Image> = Image.open('<image>.png')
    <Image>.show()
    ```

*   **Animation: 🔗** Create animations.
    ```python
    # $ pip3 install imageio
    ```

*   **Audio: 🔗**  Audio processing.
    ```python
    import wave
    #See examples
    ```

*   **Synthesizer: 🔗**  Generate audio.
    ```python
    #See examples
    ```

*   **Pygame: 🔗**  Game development.
    ```python
    # $ pip3 install pygame
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((width, height))
    ```

*   **Pandas: 🔗**  Data analysis and manipulation.
    ```python
    # $ pip3 install pandas
    import pandas as pd
    <DF> = pd.read_csv('<file>.csv')
    <DF>.plot(...)
    ```

*   **Plotly: 🔗**  Interactive visualizations.
    ```python
    # $ pip3 install plotly
    import plotly.express as px
    <Fig> = px.line(<DF>, x='Date', y='Value', color='Category')
    <Fig>.show()
    ```

---
# Comprehensive Python Cheatsheet 🔗

**Python is a versatile and widely-used programming language, known for its readability and extensive libraries, making it ideal for both beginners and experienced developers.** This cheatsheet provides a comprehensive overview of Python's core concepts, syntax, and libraries, optimized for quick reference and efficient coding.

**Key Features:**

*   **Collections:** Lists, Dictionaries, Sets, Tuples, Iterators, Generators, etc.
*   **Types:** Strings, Regular Expressions, Numbers, Datetime, and more.
*   **Syntax:** Functions, Imports, Decorators, Classes, Exceptions, etc.
*   **System:** File I/O, Command-Line Arguments, and OS commands.
*   **Data:** JSON, Pickle, CSV, SQLite, Bytes, Struct, Array, etc.
*   **Advanced:** Operator, Match Statements, Logging, Introspection, Threads, Coroutines, etc.
*   **Libraries:** Plotting, Console Apps, GUI Apps, Web Scraping, Profiling, etc.
*   **Multimedia:** NumPy, Image, Animation, Audio, Pygame, Pandas, Plotly, etc.

**[Original Repo](https://github.com/gto76/python-cheatsheet)**

---

## 1. Collections 🔗

*   **List:**  `[<el_1>, <el_2>, ...]` (Creates a list). Access/modify elements with `<list>[index]`. Methods: `.append()`, `.extend()`, `.sort()`, `.reverse()`, `.index()`, `.pop()`, `.insert()`, `.remove()`, `.clear()`
*   **Dictionary:** `{key_1: val_1, key_2: val_2, ...}` (Use `<dict>[key]` to get/set values). Methods: `.keys()`, `.values()`, `.items()`, `.get()`, `.setdefault()`, `.update()`, `.pop()`.
*   **Set:** `{<el_1>, <el_2>, ...}` (Use `set()` for an empty set). Methods: `.add()`, `.update()`, `.union()`, `.intersection()`, `.difference()`, `.symmetric_difference()`, `.issubset()`, `.issuperset()`.
*   **Tuple:** `(<el_1>, <el_2> [, ...])` (Immutable list).
*   **Range:** `range(stop)`, `range(start, stop)`, `range(start, stop, ±step)` (Immutable sequence of integers).
*   **Enumerate:** `for i, el in enumerate(<coll>, start=0): ...` (Iterates with index).
*   **Iterator:** `iter(<collection>)`, `next(<iter> [, default])` (Potentially endless stream of elements).
*   **Generator:** `yield` (Function that returns a generator).

---

## 2. Types 🔗

*   **Type:** `type(<el>)`, `isinstance(<el>, <type>)` (Everything is an object, everything has a type).
*   **String:**  Immutable sequence of characters.  Methods: `.strip()`, `.split()`, `.join()`, `.startswith()`, `.find()`, `.lower()`, `.replace()`, `.translate()`, `.chr()`, `.ord()`.
*   **Regex:**  For regular expression matching.
*   **Format:** `f'{<el_1>}, {<el_2>}'`  (String formatting).
*   **Numbers:** `int(<float/str/bool>)`, `float(<int/str/bool>)`, `complex()`, `fractions.Fraction()`, `decimal.Decimal()`.
*   **Combinatorics:**  `itertools.product()`, `itertools.permutations()`, `itertools.combinations()`.
*   **Datetime:** `date()`, `time()`, `datetime()`, `timedelta()`.

---

## 3. Syntax 🔗

*   **Function:**  `def <func_name>(<nondefault_args>): ...` (Independent block of code).
*   **Inline:** `lambda`, comprehensions, map, filter, reduce.
*   **Import:** `import <module>`, `from <module> import <obj>`.
*   **Decorator:** `@decorator_name` (Adds functionality to a function).
*   **Class:**  `class MyClass: ...` (Template for creating objects).
*   **Duck Types:** Comparable, Hashable, Sortable, Iterator, Callable, Context Manager, Iterable, Collection, Sequence.
*   **Enum:** `from enum import Enum; class <enum_name>(Enum): ...` (Class of named constants).
*   **Except:** `try...except...else...finally` (Handling exceptions).

---

## 4. System 🔗

*   **Exit:** `sys.exit()` (Exits the interpreter).
*   **Print:** `print(<el_1>, ..., sep=' ', end='\n', file=sys.stdout, flush=False)`.
*   **Input:** `<str> = input()` (Reads user input).
*   **Command-Line Arguments:**  `sys.argv` (Access command-line arguments).
*   **Open:** `open(<path>, mode='r', encoding=None, newline=None)` (Opens a file).
*   **Path:** `os.path`, `pathlib.Path` (Working with file paths).
*   **OS Commands:** `os.chdir()`, `os.mkdir()`, `shutil.copy()`, `os.rename()`, `os.remove()`, `subprocess.run()` (Operating system commands).

---

## 5. Data 🔗

*   **JSON:**  `json.dumps()`, `json.loads()` (Storing data in JSON format).
*   **Pickle:**  `pickle.dumps()`, `pickle.loads()` (Storing Python objects).
*   **CSV:** `csv.reader()`, `csv.writer()` (Reading and writing CSV files).
*   **SQLite:** `sqlite3.connect()`,  `conn.execute()`, `conn.commit()` (Working with SQLite databases).
*   **Bytes:** `bytes()`, `bytes.fromhex()`, `<bytes>.decode()` (Immutable sequence of single bytes).
*   **Struct:**  `struct.pack()`, `struct.unpack()` (Converting between Python values and C structs).
*   **Array:**  `array.array()` (Array of a predefined type).
*   **Memory View:** `memoryview()` (Sequence object that points to the memory of another bytes-like object).
*   **Deque:** `collections.deque()` (List with efficient appends and pops from either side).

---

## 6. Advanced 🔗

*   **Operator:** `import operator as op` (Functions for operator functionality).
*   **Match Statement:** `match <object/expression>: ... case <pattern>: ...` (Pattern matching).
*   **Logging:** `import logging as log` (Logging messages).
*   **Introspection:** `dir()`, `vars()`, `hasattr()`, `getattr()`, `setattr()`, `delattr()` (Examining objects).
*   **Threading:** `threading.Thread`, `Lock`, `RLock`, `Semaphore`, `Event`, `Barrier`, `ThreadPoolExecutor` (Multithreading).
*   **Coroutines:** `async def`, `await`, `asyncio.run()` (Asynchronous programming).

---

## 7. Libraries 🔗

*   **Progress Bar:** `tqdm` (Progress bar).
*   **Plot:** `matplotlib.pyplot` (Plotting data).
*   **Table:** `tabulate` (Printing tables).
*   **Console App:** `curses` (Creating console applications).
*   **GUI App:** `PySimpleGUI` (Creating GUI applications).
*   **Scraping:** `requests`, `BeautifulSoup`, `selenium` (Web scraping).
*   **Web App:** `flask` (Creating web applications).
*   **Profiling:**  `timeit`, `cProfile`, `line_profiler`, `pyinstrument`, `py-spy`, `scalene`, `memray` (Profiling code).

---

## 8. Multimedia 🔗

*   **NumPy:**  `numpy` (Numerical computing).
*   **Image:** `PIL.Image` (Image manipulation).
*   **Animation:**  `imageio` (Creating animations).
*   **Audio:** `wave`, `simpleaudio`, `pyttsx3` (Audio processing and text-to-speech).
*   **Synthesizer:** (Generating audio).
*   **Pygame:** (Game development).
*   **Pandas:** (Data analysis).
*   **Plotly:** (Interactive plotting).
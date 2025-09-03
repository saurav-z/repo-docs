# Comprehensive Python Cheatsheet 🔗

**Master Python with this comprehensive cheatsheet, packed with essential syntax, features, and libraries to supercharge your coding. Explore the original repository at [https://github.com/gto76/python-cheatsheet](https://github.com/gto76/python-cheatsheet) for more details and updates.**

## Key Features:

*   **Complete Coverage:** From basic data types to advanced concepts like coroutines and multiprocessing.
*   **Code Examples:** Clear and concise code snippets to illustrate each concept.
*   **Library Overviews:** Quick guides to popular libraries like NumPy, Pandas, and Plotly.
*   **SEO Optimized:** Content is organized with clear headings and keywords for easy navigation and searchability.
*   **Quick Reference:** Functions, methods, and syntax are easy to find and implement.

## Table of Contents:

1.  [Collections](#collections) 🔗
2.  [Types](#types) 🔗
3.  [Syntax](#syntax) 🔗
4.  [System](#system) 🔗
5.  [Data](#data) 🔗
6.  [Advanced](#advanced) 🔗
7.  [Libraries](#libraries) 🔗
8.  [Multimedia](#multimedia) 🔗
9.  [Appendix](#appendix) 🔗

## 1. Collections 🔗

*   **List** 🔗
    *   Creation, accessing elements, appending, extending, sorting, reversing, finding min/max/sum.
    *   `len()`, `count()`, `index()`, `pop()`, `insert()`, `remove()`, `clear()`.
*   **Dictionary** 🔗
    *   Creation, accessing values, `.keys()`, `.values()`, `.items()`.
    *   `.get()`, `.setdefault()`, `collections.defaultdict()`.
    *   Creating from key-value pairs, `.update()`, `.pop()`, filtering.
    *   **Counter** 🔗  Using `collections.Counter`.
*   **Set** 🔗
    *   Creation, `.add()`, `.update()`.
    *   `.union()`, `.intersection()`, `.difference()`, `.symmetric_difference()`, `.issubset()`, `.issuperset()`.
    *   `.pop()`, `.remove()`, `.discard()`.
*   **Frozen Set** 🔗  Immutable sets.
*   **Tuple** 🔗
    *   Immutable lists.
    *   **Named Tuple** 🔗  Using `collections.namedtuple`.
*   **Range** 🔗  Immutable sequence of integers.
*   **Enumerate** 🔗  Iterating with indices.
*   **Iterator** 🔗
    *   `iter()`, `next()`, creating iterators.
    *   **Itertools** 🔗  Count, repeat, cycle, chain, islice.
*   **Generator** 🔗  Functions with `yield`.

## 2. Types 🔗

*   **Type** 🔗  Everything is an object, everything has a type.  `type()`, `isinstance()`.
    *   Built-in and imported type names.
    *   **Abstract Base Classes** 🔗  `Iterable`, `Collection`, `Sequence`, `Number`, `Complex`, `Real`, `Rational`, `Integral`.
*   **String** 🔗
    *   `.strip()`, splitting, joining.
    *   `.splitlines()`,  `.join()`, substring checking, `.startswith()`, `.find()`.
    *   Case conversion, `.replace()`, `.translate()`.
    *   `chr()`, `ord()`.
    *   **Property Methods** 🔗 `.isdecimal()`, `.isdigit()`, `.isnumeric()`, `.isalnum()`, `.isprintable()`, `.isspace()`.
*   **Regex** 🔗
    *   `re.sub()`, `re.findall()`, `re.split()`, `re.search()`, `re.match()`, `re.finditer()`.
    *   **Match Object** 🔗  `.group()`, `.groups()`, `.start()`, `.end()`.
    *   **Special Sequences** 🔗  `\d`, `\w`, `\s`.
*   **Format** 🔗
    *   f-strings, `.format()`, C-style formatting.
    *   General options, strings, numbers, floats, ints.
*   **Numbers** 🔗
    *   `int()`, `float()`, `complex()`.
    *   `fractions.Fraction()`, `decimal.Decimal()`.
    *   Built-in functions:  `pow()`, `abs()`, `round()`, `min()`, `sum()`.
    *   **Math** 🔗  `floor()`, `ceil()`, `trunc()`, `pi`, `inf`, `nan`, `isnan()`, `sqrt()`, `factorial()`, trig functions, `log()`.
    *   **Statistics** 🔗  `mean()`, `median()`, `mode()`, `variance()`, `stdev()`.
    *   **Random** 🔗  `random()`, `randint()`, `uniform()`, `gauss()`, `choice()`, `shuffle()`.
    *   Hexadecimal numbers, bitwise operators.
*   **Combinatorics** 🔗  `itertools.product()`, `itertools.permutations()`, `itertools.combinations()`.
*   **Datetime** 🔗
    *   `date()`, `time()`, `datetime()`, `timedelta()`.
    *   **Now** 🔗 `.today()`, `.now()`.
    *   **Timezones** 🔗  `timezone.utc`,  `timezone()`, `dateutil.tz.tzlocal()`,  `zoneinfo.ZoneInfo()`, `.astimezone()`, `.replace()`.
    *   **Encode** 🔗 `.fromisoformat()`, `.strptime()`, `.fromordinal()`, `.fromtimestamp()`.
    *   **Decode** 🔗  `.isoformat()`, `.strftime()`, `.toordinal()`, `.timestamp()`.
    *   **Format** 🔗
    *   **Arithmetics** 🔗  Date/time operations.

## 3. Syntax 🔗

*   **Function** 🔗
    *   Defining functions.
    *   **Function Call** 🔗  Positional, keyword arguments, `*args`, `**kwargs`.
*   **Splat Operator** 🔗
    *   Unpacking arguments, packing arguments, splatting in function definitions.
*   **Inline** 🔗
    *   **Lambda** 🔗  Anonymous functions.
    *   **Comprehensions** 🔗  List, generator, set, and dictionary comprehensions.
    *   **Map, Filter, Reduce** 🔗  Using `map()`, `filter()`, `reduce()`.
    *   **Any, All** 🔗  `any()`, `all()`.
    *   **Conditional Expression** 🔗  `if ... else`.
    *   **And, Or** 🔗  Short-circuiting.
    *   **Walrus Operator** 🔗  `:=`.
    *   **Named Tuple, Enum, Dataclass** 🔗  Creating data structures.
*   **Imports** 🔗  Built-in, package, module.
*   **Closure** 🔗
    *   **Partial** 🔗  Using `functools.partial`.
    *   **Non-Local** 🔗  Modifying variables in enclosing scopes.
*   **Decorator** 🔗
    *   Debugger example, cache example, parametrized decorators.
*   **Class** 🔗
    *   Defining classes, `__init__`, `__str__`, `__repr__`,  `@classmethod`.
    *   **Subclass** 🔗  Inheritance.
    *   **Type Annotations** 🔗  Type hints.
    *   **Dataclass** 🔗  Using `@dataclass`.
    *   **Property** 🔗  Getters and setters.
    *   **Slots** 🔗  `__slots__`.
    *   **Copy** 🔗  Using `copy()`, `deepcopy()`.
*   **Duck Types** 🔗
    *   **Comparable** 🔗  `__eq__`, `__ne__`.
    *   **Hashable** 🔗  `__hash__`.
    *   **Sortable** 🔗  `__lt__`, `__gt__`, `__le__`, `__ge__`,  `functools.total_ordering`.
    *   **Iterator** 🔗  `__next__`,  `__iter__`.
    *   **Callable** 🔗  `__call__`, `callable()`.
    *   **Context Manager** 🔗  `__enter__`, `__exit__`, using `with`.
*   **Iterable Duck Types** 🔗
    *   **Iterable** 🔗  `__iter__`, `__contains__`.
    *   **Collection** 🔗  `__len__`.
    *   **Sequence** 🔗  `__getitem__`.
    *   **ABC Sequence** 🔗  `collections.abc.Sequence`.
*   **Enum** 🔗
    *   Using `enum.Enum`, enum members,  accessing members, listing members, inline creation.
*   **Exceptions** 🔗
    *   `try...except...else...finally`.
    *   Catching exceptions, re-raising exceptions, exception object attributes.
    *   **Built-in Exceptions** 🔗  List of built-in exceptions.
    *   User-defined exceptions.
*   **Exit** 🔗  `sys.exit()`.

## 4. System 🔗

*   **Print** 🔗  `print()`.
    *   **Pretty Print** 🔗  Using `pprint()`.
*   **Input** 🔗  `input()`.
*   **Command Line Arguments** 🔗  `sys.argv`, `argparse`.
    *   `ArgumentParser`.
*   **Open** 🔗  `open()`.
    *   File modes, file object methods, read/write examples, common exceptions.
*   **Paths** 🔗
    *   `os`, `glob`, `pathlib`.
    *   `os.getcwd()`, `os.path.join()`, `os.path.realpath()`, etc.
    *   `os.listdir()`, `glob.glob()`, `os.path.exists()`, etc.
    *   **DirEntry** 🔗  Using `os.scandir()`.
    *   **Path Object** 🔗  Using `pathlib.Path`.
*   **OS Commands** 🔗
    *   `os.chdir()`, `os.mkdir()`, `os.makedirs()`, copying, renaming, deleting files/directories.
    *   **Shell Commands** 🔗  Using `os.popen()`, `subprocess.run()`.

## 5. Data 🔗

*   **JSON** 🔗  `json.dumps()`, `json.loads()`, read/write to file.
*   **Pickle** 🔗  `pickle.dumps()`, `pickle.loads()`, read/write to file.
*   **CSV** 🔗  `csv.reader()`, `csv.writer()`, parameters, dialects, read/write rows.
*   **SQLite** 🔗
    *   `sqlite3.connect()`,  `.execute()`, `.fetchone()`, `.fetchall()`, `.commit()`, `.rollback()`.
    *   Placeholders, SQLAlchemy.
*   **Bytes** 🔗
    *   Bytes creation and operations.
    *   Encode, decode, read/write bytes from/to file.
*   **Struct** 🔗  Using `struct.pack()`, `struct.unpack()`, format strings.
*   **Array** 🔗  Using `array.array()`.
*   **Memory View** 🔗  Using `memoryview()`.
*   **Deque** 🔗  Using `collections.deque()`.

## 6. Advanced 🔗

*   **Operator** 🔗  Using `operator` module.
*   **Match Statement** 🔗  Pattern matching.
*   **Logging** 🔗
    *   `logging.basicConfig()`,  log levels, formatters, handlers, loggers.
    *   Setup, file handling, message formatting.
*   **Introspection** 🔗
    *   `dir()`, `vars()`, `globals()`, `hasattr()`, `getattr()`, `setattr()`, `delattr()`.
    *   `inspect.signature()`.
*   **Threading** 🔗
    *   `threading.Thread`, `threading.Lock`, `threading.RLock`, `threading.Semaphore`, `threading.Event`, `threading.Barrier`.
    *   `concurrent.futures.ThreadPoolExecutor`,  `as_completed()`, `submit()`, `map()`.
    *   Queue
*   **Coroutines** 🔗
    *   `asyncio`, `async def`, `await`,  `aio.create_task()`, `aio.gather()`, `aio.wait()`,  `aio.as_completed()`.

## 7. Libraries 🔗

*   **Progress Bar** 🔗  Using `tqdm`.
*   **Plot** 🔗  Using `matplotlib.pyplot`.
*   **Table** 🔗  Using `tabulate`.
*   **Console App** 🔗  `curses`.
*   **GUI App** 🔗  Using `PySimpleGUI`.
*   **Scraping** 🔗
    *   Using `requests`, `BeautifulSoup`.
    *   **Selenium** 🔗  Using `selenium.webdriver`.
    *   XPath
*   **Web App** 🔗  Using `flask`.

## 8. Multimedia 🔗

*   **NumPy** 🔗
    *   Creating arrays, reshaping, flattening, transposing.
    *   Copying, mathematical operations, aggregating.
    *   Concatenation, stacking, tiling.
    *   **Indexing** 🔗  Array indexing and slicing.
    *   **Broadcasting** 🔗  NumPy broadcasting rules.
*   **Image** 🔗
    *   Using `PIL.Image`.
    *   Creating, opening, converting, saving, and showing images.
    *   Pixel manipulation, filtering, and image enhancement.
    *   **Modes** 🔗  Image modes.
    *   **Image Draw** 🔗 Drawing shapes, text and more.
*   **Animation** 🔗  Using `imageio`.
*   **Audio** 🔗  Using `wave`.
    *   Opening, reading, and writing WAV files.
    *   Sample values.
    *   Examples.
    *   Text to Speech
*   **Synthesizer** 🔗  Using `simpleaudio`.
*   **Pygame** 🔗
    *   Opens window, draws square and uses key presses
    *   **Rect** 🔗  Using `pygame.Rect`.
    *   **Surface** 🔗 Using `pygame.Surface`.
    *   Images, sound

## 9. Appendix 🔗

*   **Cython** 🔗  Writing fast C code.
*   **Virtual Environments** 🔗  Creating and using virtual environments.
*   **Basic Script Template** 🔗  Template for Python scripts.
*   **Index** 🔗  Index of all the topics.
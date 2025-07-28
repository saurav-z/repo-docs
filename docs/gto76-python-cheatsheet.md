html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Python Cheatsheet: Your Ultimate Python Guide</title>
    <meta name="description" content="Master Python with this comprehensive cheatsheet covering collections, types, syntax, system commands, data handling, advanced techniques, libraries, and multimedia. Learn Python fast!">
    <meta name="keywords" content="Python, cheatsheet, tutorial, reference, collections, types, syntax, libraries, data science, programming">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }

        h1, h2, h3 {
            color: #333;
        }

        ul {
            list-style-type: disc;
            margin-left: 20px;
        }

        code {
            font-family: monospace;
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }

        a {
            color: #007bff;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        .key-features {
            margin-bottom: 20px;
        }

        .section {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>

    <h1>Comprehensive Python Cheatsheet: Your Go-To Python Resource</h1>
    <p><strong>Unlock your Python potential with this all-encompassing cheatsheet.</strong> Access the original repository on GitHub: <a href="https://github.com/gto76/python-cheatsheet">gto76/python-cheatsheet</a></p>

    <div class="key-features">
        <h2>Key Features</h2>
        <ul>
            <li><strong>Comprehensive Coverage:</strong> Includes collections, types, syntax, system commands, data handling, advanced techniques, libraries, and multimedia.</li>
            <li><strong>Easy-to-Read Format:</strong> Organized with clear headings, bullet points, and code examples for quick understanding.</li>
            <li><strong>SEO-Optimized:</strong> Targeted keywords and descriptions to improve searchability.</li>
            <li><strong>Practical Examples:</strong> Illustrates concepts with real-world use cases.</li>
            <li><strong>Up-to-Date:</strong>  Maintained and updated to reflect the latest Python features.</li>
        </ul>
    </div>

    <div class="section">
        <h2>1. Collections</h2>
        <ul>
            <li><a href="#list">List</a> </li>
            <li><a href="#dictionary">Dictionary</a></li>
            <li><a href="#set">Set</a></li>
            <li><a href="#tuple">Tuple</a></li>
            <li><a href="#range">Range</a></li>
            <li><a href="#enumerate">Enumerate</a></li>
            <li><a href="#iterator">Iterator</a></li>
            <li><a href="#generator">Generator</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>2. Types</h2>
        <ul>
            <li><a href="#type">Type</a></li>
            <li><a href="#string">String</a></li>
            <li><a href="#regex">Regular Expression (Regex)</a></li>
            <li><a href="#format">Format</a></li>
            <li><a href="#numbers">Numbers</a></li>
            <li><a href="#combinatorics">Combinatorics</a></li>
            <li><a href="#datetime">Datetime</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>3. Syntax</h2>
        <ul>
            <li><a href="#function">Function</a></li>
            <li><a href="#inline">Inline</a></li>
            <li><a href="#imports">Imports</a></li>
            <li><a href="#decorator">Decorator</a></li>
            <li><a href="#class">Class</a></li>
            <li><a href="#duck-types">Duck Types</a></li>
            <li><a href="#enum">Enum</a></li>
            <li><a href="#exceptions">Exceptions</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>4. System</h2>
        <ul>
            <li><a href="#exit">Exit</a></li>
            <li><a href="#print">Print</a></li>
            <li><a href="#input">Input</a></li>
            <li><a href="#command-line-arguments">Command Line Arguments</a></li>
            <li><a href="#open">Open</a></li>
            <li><a href="#paths">Paths</a></li>
            <li><a href="#os-commands">OS Commands</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>5. Data</h2>
        <ul>
            <li><a href="#json">JSON</a></li>
            <li><a href="#pickle">Pickle</a></li>
            <li><a href="#csv">CSV</a></li>
            <li><a href="#sqlite">SQLite</a></li>
            <li><a href="#bytes">Bytes</a></li>
            <li><a href="#struct">Struct</a></li>
            <li><a href="#array">Array</a></li>
            <li><a href="#memory-view">Memory View</a></li>
            <li><a href="#deque">Deque</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>6. Advanced</h2>
        <ul>
            <li><a href="#operator">Operator</a></li>
            <li><a href="#match-statement">Match Statement</a></li>
            <li><a href="#logging">Logging</a></li>
            <li><a href="#introspection">Introspection</a></li>
            <li><a href="#threading">Threading</a></li>
            <li><a href="#coroutines">Coroutines</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>7. Libraries</h2>
        <ul>
            <li><a href="#progress-bar">Progress Bar</a></li>
            <li><a href="#plot">Plot</a></li>
            <li><a href="#table">Table</a></li>
            <li><a href="#console-app">Console App</a></li>
            <li><a href="#gui-app">GUI App</a></li>
            <li><a href="#scraping">Scraping</a></li>
            <li><a href="#web-app">Web App</a></li>
            <li><a href="#profiling">Profiling</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>8. Multimedia</h2>
        <ul>
            <li><a href="#numpy">NumPy</a></li>
            <li><a href="#image">Image</a></li>
            <li><a href="#animation">Animation</a></li>
            <li><a href="#audio">Audio</a></li>
            <li><a href="#synthesizer">Synthesizer</a></li>
            <li><a href="#pygame">Pygame</a></li>
            <li><a href="#pandas">Pandas</a></li>
            <li><a href="#plotly">Plotly</a></li>
        </ul>
    </div>

    <div class="section">
        <h2><a id="main"></a>Main</h2>
        <pre>
            <code class="language-python">
                if __name__ == '__main__':      # Skips next line if file was imported.
                    main()                      # Runs `def main(): ...` function.
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="list"></a>List</h2>
        <pre>
            <code class="language-python">
                <list> = [<el_1>, <el_2>, ...]  # Creates a list object. Also list(<collection>).
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <el>   = <list>[index]          # First index is 0. Last -1. Allows assignments.
                <list> = <list>[<slice>]        # Also <list>[from_inclusive : to_exclusive : ±step].
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <list>.append(<el>)             # Appends element to the end. Also <list> += [<el>].
                <list>.extend(<collection>)     # Appends elements to the end. Also <list> += <coll>.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <list>.sort()                   # Sorts the elements in ascending order.
                <list>.reverse()                # Reverses the order of list's elements.
                <list> = sorted(<collection>)   # Returns a new list with sorted elements.
                <iter> = reversed(<list>)       # Returns reversed iterator of elements.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <el>  = max(<collection>)       # Returns largest element. Also min(<el_1>, ...).
                <num> = sum(<collection>)       # Returns sum of elements. Also math.prod(<coll>).
            </code>
        </pre>
        <pre>
            <code class="language-python">
                elementwise_sum  = [sum(pair) for pair in zip(list_a, list_b)]
                sorted_by_second = sorted(<collection>, key=lambda el: el[1])
                sorted_by_both   = sorted(<collection>, key=lambda el: (el[1], el[0]))
                flatter_list     = list(itertools.chain.from_iterable(<list>))
            </code>
        </pre>
        <ul>
            <li><strong>For details about sort(), sorted(), min() and max() see <a href="#sortable">Sortable</a>.</strong></li>
            <li><strong>Module <a href="#operator">operator</a> has function itemgetter() that can replace listed <a href="#lambda">lambdas</a>.</strong></li>
            <li><strong>This text uses the term collection instead of iterable. For rationale see <a href="#collection">Collection</a>.</strong></li>
        </ul>
        <pre>
            <code class="language-python">
                <int> = len(<list>)             # Returns number of items. Also works on dict, set and string.
                <int> = <list>.count(<el>)      # Returns number of occurrences. Also `if <el> in <coll>: ...`.
                <int> = <list>.index(<el>)      # Returns index of the first occurrence or raises ValueError.
                <el>  = <list>.pop()            # Removes and returns item from the end or at index if passed.
                <list>.insert(<int>, <el>)      # Inserts item at passed index and moves the rest to the right.
                <list>.remove(<el>)             # Removes first occurrence of the item or raises ValueError.
                <list>.clear()                  # Removes all list's items. Also works on dictionary and set.
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="dictionary"></a>Dictionary</h2>
        <pre>
            <code class="language-python">
                <dict> = {key_1: val_1, key_2: val_2, ...}      # Use `<dict>[key]` to get or set the value.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <view> = <dict>.keys()                          # Collection of keys that reflects changes.
                <view> = <dict>.values()                        # Collection of values that reflects changes.
                <view> = <dict>.items()                         # Coll. of key-value tuples that reflects chgs.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                value  = <dict>.get(key, default=None)          # Returns default argument if key is missing.
                value  = <dict>.setdefault(key, default=None)   # Returns and writes default if key is missing.
                <dict> = collections.defaultdict(<type>)        # Returns a dict with default value `<type>()`.
                <dict> = collections.defaultdict(lambda: 1)     # Returns a dict with default value 1.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <dict> = dict(<collection>)                     # Creates a dict from coll. of key-value pairs.
                <dict> = dict(zip(keys, values))                # Creates a dict from two collections.
                <dict> = dict.fromkeys(keys [, value])          # Creates a dict from collection of keys.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <dict>.update(<dict>)                           # Adds items. Replaces ones with matching keys.
                value = <dict>.pop(key)                         # Removes item or raises KeyError if missing.
                {k for k, v in <dict>.items() if v == value}    # Returns set of keys that point to the value.
                {k: v for k, v in <dict>.items() if k in keys}  # Filters the dictionary by specified keys.
            </code>
        </pre>
        <h3>Counter</h3>
        <pre>
            <code class="language-python">
                >>> from collections import Counter
                >>> counter = Counter(['blue', 'blue', 'blue', 'red', 'red'])
                >>> counter['yellow'] += 1
                >>> print(counter.most_common())
                [('blue', 3), ('red', 2), ('yellow', 1)]
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="set"></a>Set</h2>
        <pre>
            <code class="language-python">
                <set> = {<el_1>, <el_2>, ...}                   # Use `set()` for empty set.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <set>.add(<el>)                                 # Or: <set> |= {<el>}
                <set>.update(<collection> [, ...])              # Or: <set> |= <set>
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <set>  = <set>.union(<coll.>)                   # Or: <set> | <set>
                <set>  = <set>.intersection(<coll.>)            # Or: <set> & <set>
                <set>  = <set>.difference(<coll.>)              # Or: <set> - <set>
                <set>  = <set>.symmetric_difference(<coll.>)    # Or: <set> ^ <set>
                <bool> = <set>.issubset(<coll.>)                # Or: <set> <= <set>
                <bool> = <set>.issuperset(<coll.>)              # Or: <set> >= <set>
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <el> = <set>.pop()                              # Raises KeyError if empty.
                <set>.remove(<el>)                              # Raises KeyError if missing.
                <set>.discard(<el>)                             # Doesn't raise an error.
            </code>
        </pre>
        <h3>Frozen Set</h3>
        <ul>
            <li><strong>Is immutable and hashable.</strong></li>
            <li><strong>That means it can be used as a key in a dictionary or as an element in a set.</strong></li>
        </ul>
        <pre>
            <code class="language-python">
                <frozenset> = frozenset(<collection>)
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="tuple"></a>Tuple</h2>
        <p><strong>Tuple is an immutable and hashable list.</strong></p>
        <pre>
            <code class="language-python">
                <tuple> = ()                               # Empty tuple.
                <tuple> = (<el>,)                          # Or: <el>,
                <tuple> = (<el_1>, <el_2> [, ...])         # Or: <el_1>, <el_2> [, ...]
            </code>
        </pre>
        <h3>Named Tuple</h3>
        <p><strong>Tuple's subclass with named elements.</strong></p>
        <pre>
            <code class="language-python">
                >>> from collections import namedtuple
                >>> Point = namedtuple('Point', 'x y')
                >>> p = Point(1, y=2)
                >>> print(p)
                Point(x=1, y=2)
                >>> p.x, p[1]
                (1, 2)
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="range"></a>Range</h2>
        <p><strong>Immutable and hashable sequence of integers.</strong></p>
        <pre>
            <code class="language-python">
                <range> = range(stop)                      # I.e. range(to_exclusive).
                <range> = range(start, stop)               # I.e. range(from_inclusive, to_exclusive).
                <range> = range(start, stop, ±step)        # I.e. range(from_inclusive, to_exclusive, ±step).
            </code>
        </pre>
        <pre>
            <code class="language-python">
                >>> [i for i in range(3)]
                [0, 1, 2]
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="enumerate"></a>Enumerate</h2>
        <pre>
            <code class="language-python">
                for i, el in enumerate(<coll>, start=0):   # Returns next element and its index on each pass.
                    ...
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="iterator"></a>Iterator</h2>
        <p><strong>Potentially endless stream of elements.</strong></p>
        <pre>
            <code class="language-python">
                <iter> = iter(<collection>)                # `iter(<iter>)` returns unmodified iterator.
                <iter> = iter(<function>, to_exclusive)    # A sequence of return values until 'to_exclusive'.
                <el>   = next(<iter> [, default])          # Raises StopIteration or returns 'default' on end.
                <list> = list(<iter>)                      # Returns a list of iterator's remaining elements.
            </code>
        </pre>
        <h3>Itertools</h3>
        <pre>
            <code class="language-python">
                import itertools as it
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <iter> = it.count(start=0, step=1)         # Returns updated value endlessly. Accepts floats.
                <iter> = it.repeat(<el> [, times])         # Returns element endlessly or 'times' times.
                <iter> = it.cycle(<collection>)            # Repeats the passed sequence of elements endlessly.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <iter> = it.chain(<coll>, <coll> [, ...])  # Empties collections in order (only figuratively).
                <iter> = it.chain.from_iterable(<coll>)    # Empties collections inside a collection in order.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <iter> = it.islice(<coll>, to_exclusive)   # Only returns first 'to_exclusive' elements.
                <iter> = it.islice(<coll>, from_inc, …)    # `to_exclusive, +step_size`. Indices can be None.
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="generator"></a>Generator</h2>
        <ul>
            <li><strong>Any function that contains a yield statement returns a generator.</strong></li>
            <li><strong>Generators and iterators are interchangeable.</strong></li>
        </ul>
        <pre>
            <code class="language-python">
                def count(start, step):
                    while True:
                        yield start
                        start += step
            </code>
        </pre>
        <pre>
            <code class="language-python">
                >>> counter = count(10, 2)
                >>> next(counter), next(counter), next(counter)
                (10, 12, 14)
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="type"></a>Type</h2>
        <ul>
            <li><strong>Everything is an object.</strong></li>
            <li><strong>Every object has a type.</strong></li>
            <li><strong>Type and class are synonymous.</strong></li>
        </ul>
        <pre>
            <code class="language-python">
                <type> = type(<el>)                          # Or: <el>.__class__
                <bool> = isinstance(<el>, <type>)            # Or: issubclass(type(<el>), <type>)
            </code>
        </pre>
        <pre>
            <code class="language-python">
                >>> type('a'), 'a'.__class__, str
                (<class 'str'>, <class 'str'>, <class 'str'>)
            </code>
        </pre>
        <h4>Some types do not have built-in names, so they must be imported:</h4>
        <pre>
            <code class="language-python">
                from types import FunctionType, MethodType, LambdaType, GeneratorType, ModuleType
            </code>
        </pre>
        <h3>Abstract Base Classes</h3>
        <p><strong>Each abstract base class specifies a set of virtual subclasses. These classes are then recognized by isinstance() and issubclass() as subclasses of the ABC, although they are really not. ABC can also manually decide whether or not a specific class is its virtual subclass, usually based on which methods the class has implemented. For instance, Iterable ABC looks for method iter(), while Collection ABC looks for iter(), contains() and len().</strong></p>
        <pre>
            <code class="language-python">
                >>> from collections.abc import Iterable, Collection, Sequence
                >>> isinstance([1, 2, 3], Iterable)
                True
            </code>
        </pre>
        <pre>
            <code class="language-text">
                +------------------+------------+------------+------------+
                |                  |  Iterable  | Collection |  Sequence  |
                +------------------+------------+------------+------------+
                | list, range, str |    yes     |    yes     |    yes     |
                | dict, set        |    yes     |    yes     |            |
                | iter             |    yes     |            |            |
                +------------------+------------+------------+------------+
            </code>
        </pre>
        <pre>
            <code class="language-python">
                >>> from numbers import Number, Complex, Real, Rational, Integral
                >>> isinstance(123, Number)
                True
            </code>
        </pre>
        <pre>
            <code class="language-text">
                +--------------------+----------+----------+----------+----------+----------+
                |                    |  Number  |  Complex |   Real   | Rational | Integral |
                +--------------------+----------+----------+----------+----------+----------+
                | int                |   yes    |   yes    |   yes    |   yes    |   yes    |
                | fractions.Fraction |   yes    |   yes    |   yes    |   yes    |          |
                | float              |   yes    |   yes    |   yes    |          |          |
                | complex            |   yes    |   yes    |          |          |          |
                | decimal.Decimal    |   yes    |          |          |          |          |
                +--------------------+----------+----------+----------+----------+----------+
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="string"></a>String</h2>
        <p><strong>Immutable sequence of characters.</strong></p>
        <pre>
            <code class="language-python">
                <str>  = <str>.strip()                       # Strips all whitespace characters from both ends.
                <str>  = <str>.strip('<chars>')              # Strips passed characters. Also lstrip/rstrip().
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <list> = <str>.split()                       # Splits on one or more whitespace characters.
                <list> = <str>.split(sep=None, maxsplit=-1)  # Splits on 'sep' string at most 'maxsplit' times.
                <list> = <str>.splitlines(keepends=False)    # On [\n\r\f\v\x1c-\x1e\x85\u2028\u2029] and \r\n.
                <str>  = <str>.join(<coll_of_strings>)       # Joins elements by using string as a separator.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <bool> = <sub_str> in <str>                  # Checks if string contains the substring.
                <bool> = <str>.startswith(<sub_str>)         # Pass tuple of strings for multiple options.
                <int>  = <str>.find(<sub_str>)               # Returns start index of the first match or -1.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <str>  = <str>.lower()                       # Lowers the case. Also upper/capitalize/title().
                <str>  = <str>.casefold()                    # Same, but converts ẞ/ß to ss, Σ/ς to σ, etc.
                <str>  = <str>.replace(old, new [, count])   # Replaces 'old' with 'new' at most 'count' times.
                <str>  = <str>.translate(<table>)            # Use `str.maketrans(<dict>)` to generate table.
            </code>
        </pre>
        <pre>
            <code class="language-python">
                <str>  = chr(<int>)                          # Converts passed integer to Unicode character.
                <int>  = ord(<str>)                          # Converts passed Unicode character to integer.
            </code>
        </pre>
        <ul>
            <li><strong>Use `'unicodedata.normalize("NFC", <str>)'` on strings like `'Motörhead'` before comparing them to other strings, because `'ö'` can be stored as one or two characters.</strong></li>
            <li><strong>`'NFC'` converts such characters to a single character, while `'NFD'` converts them to two.</strong></li>
        </ul>
        <h3>Property Methods</h3>
        <pre>
            <code class="language-python">
                <bool> = <str>.isdecimal()                   # Checks for [0-9]. Also [०-९] and [٠-٩].
                <bool> = <str>.isdigit()                     # Checks for [²³¹…] and isdecimal().
                <bool> = <str>.isnumeric()                   # Checks for [¼½¾…], [零〇一…] and isdigit().
                <bool> = <str>.isalnum()                     # Checks for [a-zA-Z…] and isnumeric().
                <bool> = <str>.isprintable()                 # Checks for [ !#$%…] and isalnum().
                <bool> = <str>.isspace()                     # Checks for [ \t\n\r\f\v\x1c-\x1f\x85…].
            </code>
        </pre>
    </div>

    <div class="section">
        <h2><a id="regex"></a>Regex</h2>
        <p><strong>Functions for regular expression matching.</strong></p>
        <pre>
            <code class="language-python">
                import re
                <str>   = re.sub(r'<regex>', new, text, count=0)  # Substitutes all occurrences with 'new'.
                <list>  = re.findall(r'<regex>', text)            # Returns all occurrences of the pattern.
                <list>  = re.split(r'<regex>', text, maxsplit=0)  # Add brackets around regex to keep matches.
                <Match> = re.search(r'<regex>', text)             # First occurrence of the pattern or None.
                <Match> = re.match(r'<regex>', text)              # Searches only at the beginning of the text.
                <iter>  = re.finditer(r'<regex>', text)           # Returns all occurrences as Match objects.
            </code>
        </pre>
        <ul>
            <li><strong>Raw string literals do not interpret escape sequences, thus enabling us to use regex-specific escape sequences that cause SyntaxWarning in normal string literals (since 3.12).</strong></li>
            <li><strong>Argument 'new' of re.sub() can be a function that accepts Match object and returns a str.</strong></li>
            <li><strong>Argument `'flags=re.IGNORECASE'` can be used with all listed regex functions.</strong></li>
            <li><strong>Argument `'flags=re.MULTILINE'` makes `'^'` and `'$'` match the start/end of each line.</strong></li>
            <li><strong>Argument `'flags=re.DOTALL'` makes `'.'` also accept the `'\n'` (besides all other chars).</strong></li>
            <li><strong>`'re.compile(<regex>)'` returns a Pattern object with methods sub(), findall(), etc.</strong></li>
        </ul>
        <h3>Match Object</h3>
        <pre>
            <code class="language-python">
                <str>   = <Match>.group()                         # Returns the whole match. Also group(0).
                <str>   = <Match>.group(1)                        # Returns part inside the first brackets.
                <tuple> = <Match>.groups()                        # Returns all bracketed parts as strings.
                <int>   = <Match>.start()                         # Returns start index of the whole match.
                <int>   = <Match>.end()                           # Returns its exclusive end index.
            </code>
        </pre>
        <h3>Special Sequences</h3>
        <pre>
            <code class="language-python">
                '\d' == '[0-9]'                                   # Also [०-९…]. Matches a decimal character.
                '\w' == '[a-zA-Z0-9_]'                            # Also [ª²³…]. Matches an alphanumeric or _.
                '\s' == '[ \t\n\r\f\v]'                           # Also [\x1c-\x1f…]. Matches a whitespace.
            </code>
        </pre>
        <ul>
            <li><strong>By default, decimal characters and alphanumerics from all alphabets are matched unless `'flags=re.ASCII'` is used. It restricts special sequence matches to the first 128 Unicode characters and also prevents `'\s'` from accepting `'\x1c'`, `'\x1d'`, `'\x1e'` and `'\x1f'` (non-printable characters that divide text into files, tables, rows and fields, respectively).</strong></li>
            <li><strong>Use a capital letter for negation (all non-ASCII characters will be matched when used in combination with ASCII flag).</strong></li>
        </ul>
    </div>

    <div class="section">
        <h2><a id="format"></a>Format</h2>
        <pre>
            <code class="language-python">
                <str> = f'{<el_1>}, {<el_2>}'            # Curly braces can also contain expressions.
                <str> = '{}, {}'.format(<el_1>, <el_2>)  # Same as '{0}, {a}'.format(<el_1>, a=<el_2>).
                <str> = '%s, %s' % (<el_1>, <el_2>)      # Redundant and inferior C-style formatting.
            </code>
        </pre>
        <h3>Example</h3>
        <pre>
            <code class="language-python">
                >>> Person = collections.namedtuple('Person', 'name height')
                >>> person = Person('Jean-Luc', 187)
                >>> f'{person.name} is {person.height / 100} meters tall.'
                'Jean-Luc is 1.87 meters tall.'
            </code>
        </pre>
        <h3>General Options</h3>
        <pre>
            <code class="language-python">
                {<el>:<10}                               # '<el>      '
                {<el>:^10}                               # '   <el>   '
                {<el>:>10}                               # '      <el>'
                {<el>:.<10}                              # '<el>......'
                {<el>:0}                                 # '<el>'
            </code>
        </pre>
        <ul>
            <li><strong>Objects are rendered by calling the `'format(<el>, "<options>")'` function.</strong></li>
            <li><strong>Options inside curly braces can be generated dynamically: `f'{<el>:{<str/int>}[…]}'`.</strong></li>
            <li><strong>Adding `'='` to the expression prepends it to the output: `f'{1+1=}'` returns `'1+1=2'`.</strong></li>
            <li><strong>Adding `'!r'` to the expression converts object to string by calling its [repr()](#class) method.</strong></li>
        </ul>
        <h3>Strings</h3>
        <pre>
            <code class="language-python">
                {'abcde':10}                             # 'abcde     '
                {'abcde':10.3}                           # 'abc       '
                {'abcde':.3}                             # 'abc'
                {'abcde'!r:10}                           # "'abcde'   "
            </code>
        </pre>
        <h3>Numbers</h3>
        <pre>
            <code class="language-python">
                {123456:10}                              # '    123456'
                {123456:10,}                             # '   123,456'
                {123456:10_}                             # '   123_456'
                {123456:+10}                             # '   +123456'
                {123456:=+10}                            # '+   123456'
                {123456: }                               # ' 123456'
                {-123456: }                              # '-123456'
            </code>
        </pre>
        <h3>Floats</h3>
        <pre>
            <code class="language-python">
                {1.23456:10.3}                           # '      1.23'
                {1.23456:10.3f}                          # '     1.235'
                {1.23456:10.3e}                          # ' 1.235e+00'
                {1.23456:10.3%}                          # '  123.456%'
            </code>
        </pre>
        <h4>Comparison of presentation types:</h4>
        <pre>
            <code class
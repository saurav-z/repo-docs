# Arrow: Elegant Date and Time Handling for Python

**Simplify your Python date and time operations with Arrow, a user-friendly library that makes working with dates, times, and timestamps a breeze.**  Get started with Arrow at [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow).

## Key Features of Arrow

Arrow offers a more intuitive and feature-rich approach to date and time manipulation in Python:

*   **Simplified Date and Time Creation:** Easily create Arrow objects from various input formats.
*   **Timezone Awareness:**  Built-in support for timezones and UTC by default.
*   **Human-Friendly Formatting & Parsing:**  Format and parse dates and times effortlessly.
*   **Date and Time Manipulation:**  Effortlessly shift, add, and subtract time intervals.
*   **ISO 8601 Support:**  Robust support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamlessly convert between timezones.
*   **Date and Time Span Generation:** Generate time spans, ranges, floors, and ceilings.
*   **Humanization:**  Convert dates and times into human-readable strings.
*   **Extensible:** Extend Arrow with your own custom types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the complexities of Python's built-in `datetime` and related modules by:

*   **Reducing Complexity:** Fewer modules and types to juggle.
*   **Simplifying Timezone Handling:** Makes timezone conversions and UTC handling straightforward.
*   **Enhancing Functionality:** Offers features like ISO 8601 parsing and humanization that are missing in the standard library.

## Quick Start

### Installation

Install Arrow using pip:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object
arrow.get('2013-05-11T21:23:58.970460+07:00')
# => <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC now
utc = arrow.utcnow()
# => <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
# => <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
# => <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get a timestamp
local.timestamp()
# => 1368303838.970460

# Format the date
local.format()
# => '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# => '2013-05-11 13:23:58 -07:00'

# Humanize the date
local.humanize()
# => 'an hour ago'
local.humanize(locale='ko-kr')
# => '한시간 전'
```

## Documentation

For detailed information and guides, visit the official documentation: [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions! If you'd like to contribute to Arrow:

1.  Find an issue or feature to work on in the [issue tracker](https://github.com/arrow-py/arrow/issues).  Look for issues tagged with "good first issue".
2.  Fork the repository.
3.  Create a branch and make your changes.
4.  Add tests to verify your code.
5.  Run tests with `tox` or the specified `make` commands.
6.  Submit a pull request.

For any questions, use the [discussions](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

If you find Arrow valuable, consider supporting the project through [Open Collective](https://opencollective.com/arrow) to help with its ongoing development and maintenance.
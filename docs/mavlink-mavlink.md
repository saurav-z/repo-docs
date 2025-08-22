# MAVLink: Lightweight Communication for Drones and Ground Control Stations

MAVLink is a versatile, open-source message protocol that enables seamless communication between drones, ground control stations, and other unmanned systems. You can find the original project on GitHub: [https://github.com/mavlink/mavlink](https://github.com/mavlink/mavlink)

## Key Features of MAVLink

*   **Lightweight and Efficient:** Optimized for resource-constrained systems, making it ideal for applications with limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Easy to integrate into various projects with minimal dependencies.
*   **Cross-Platform Support:** Compatible with a wide range of programming languages, including C, Python, and others (see supported languages [here](https://mavlink.io/en/#supported_languages)).
*   **XML-Based Message Definition:** Uses XML files to define message sets ("dialects"), ensuring clear and maintainable communication protocols.
*   **Extensive Documentation:** Comprehensive resources available for developers, including tutorials, examples, and API references ([mavlink.io](https://mavlink.io/en/)).
*   **Field-Proven:** Widely adopted and deployed in numerous products, facilitating interoperability between components from different manufacturers.

## Getting Started with MAVLink

Here's a quick guide to generate C headers and incorporate MAVLink into your projects:

### Prerequisites

*   Ubuntu LTS 20.04 or 22.04
*   Python 3 and pip

### Installation

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Dependencies:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

### Generate C Headers

Use the `mavgen` tool to generate C headers for a specific dialect (e.g., `common.xml`):

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

### Integrating with CMake

1.  **Install Headers:**

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Find and Link in your `CMakeLists.txt`:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Configure CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

For detailed instructions and examples, refer to the [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/) documentation.  Also, check out [examples/c](examples/c) for a complete working example.

## Further Resources

*   **Documentation/Website:** [https://mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [https://mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [https://mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [https://mavlink.io/en/#license](https://mavlink.io/en/#license)
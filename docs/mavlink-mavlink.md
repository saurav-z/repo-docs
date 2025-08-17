# MAVLink: Lightweight Communication for Drones and Ground Control

**MAVLink** is the leading open-source, header-only message library enabling seamless communication between drones, ground control stations, and other unmanned aerial vehicle (UAV) components. ([See the original repository](https://github.com/mavlink/mavlink))

## Key Features of MAVLink

*   **Lightweight and Efficient:** Designed for resource-constrained systems, optimized for limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Simplifies integration and reduces build times.
*   **Cross-Platform Support:** Offers message definitions for various systems ("dialects") defined in XML and generated code in multiple [supported languages](https://mavlink.io/en/#supported_languages)
*   **Field-Proven:** Widely deployed in numerous products, ensuring reliable interoperability between components from different manufacturers.
*   **Extensible:** The message definitions, which are specified in XML files, make it easy to extend MAVLink with custom messages.

## Getting Started with MAVLink

### 1. Prerequisites

Ensure you have the following installed:

*   `python3`
*   `pip`

### 2. Clone the Repository

```bash
git clone https://github.com/mavlink/mavlink.git --recursive
cd mavlink
```

### 3. Install Python Dependencies

```bash
python3 -m pip install -r pymavlink/requirements.txt
```

### 4. Generate C Headers (Example)

This example generates C headers from `message_definitions/v1.0/common.xml`

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

### 5. Integrate with CMake (Example)

```cmake
# Install MAVLink headers locally
cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
cmake --build build --target install

# Use find_package in your CMakeLists.txt
find_package(MAVLink REQUIRED)

add_executable(my_program my_program.c)

target_link_libraries(my_program PRIVATE MAVLink::mavlink)

# Build your project, pointing to the local install
cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
```

For comprehensive instructions, refer to:

*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)

## Further Resources

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
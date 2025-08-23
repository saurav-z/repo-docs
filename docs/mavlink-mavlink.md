# MAVLink: The Lightweight Communication Protocol for Drones and Robotics

MAVLink is a powerful, yet simple, header-only message library designed for seamless communication between drones, ground control stations, and other robotic systems. ([Original Repo](https://github.com/mavlink/mavlink))

[![Build Status](https://github.com/mavlink/mavlink/workflows/Test%20and%20deploy/badge.svg)](https://github.com/mavlink/mavlink/actions?query=branch%3Amaster)

## Key Features of MAVLink:

*   **Lightweight & Efficient:** Optimized for resource-constrained systems, perfect for applications with limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Easy to integrate, reducing build times and dependencies.
*   **Cross-Platform Compatibility:** Supports multiple programming languages and operating systems.
*   **XML-Based Message Definitions:**  Uses XML files to define message sets ("dialects"), enabling flexible and extensible communication.
*   **Field-Proven:** Deployed in numerous products and serves as a reliable interoperability interface.
*   **Comprehensive Tooling:** Offers Python scripts for code generation, examples, and utilities.
*   **Interoperability:** Enables seamless communication between components from different manufacturers.

## Getting Started

### Generate C Headers (Ubuntu Example)

1.  **Install Dependencies:**
    ```bash
    sudo apt install python3-pip
    ```
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```
3.  **Install Python Requirements:**
    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```
4.  **Generate C Headers (Example):**
    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

### Using with CMake

1.  **Install Headers Locally:**
    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```
2.  **Find the Package in CMakeLists.txt:**
    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```
3.  **Pass the Install Directory to CMake:**
    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    For a full example, refer to `examples/c`.

*Note:  Even though `target_link_libraries` is used in CMake, MAVLink is header-only, so it does not actually "link".*

### Further Resources

*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)

## Key Links

*   **Documentation/Website:** [https://mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [https://mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [https://mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [https://mavlink.io/en/#license](https://mavlink.io/en/#license)
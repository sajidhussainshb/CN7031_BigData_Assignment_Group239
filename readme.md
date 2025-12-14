# Spark Data Processing Framework - Technical Contribution



## Assignment Overview
This repository contains the implementation of **4 non-similar tasks** from the Spark Data Processing framework assignment, demonstrating advanced PySpark capabilities in text processing, window functions, configuration tuning, and persistence strategies.

## Tasks Completed

### ✅ Task 4: Advanced Text Feature Extraction Without UDFs
**File:** `task4_advanced_regex_extraction.py`

**Objective:** Process unstructured text data using complex REGEX patterns without User-Defined Functions (UDFs).

**Key Features:**
- Extracts 9+ features from unstructured text using built-in Spark functions
- Implements complex REGEX patterns for email, phone, date, credit card, IP address detection
- Demonstrates data masking for security (credit card anonymization)
- Provides extraction statistics and success rates

**Technical Highlights:**
- No UDFs used - 100% Spark built-in functions (`regexp_extract`, `regexp_replace`, `rlike`)
- Complex pattern matching with alternatives and groups
- Performance-optimized using native JVM functions

### ✅ Task 5: Window Function Optimization
**File:** `task5_window_functions.py`

**Objective:** Implement non-trivial window functions and analyze execution plans and data shuffling.

**Key Features:**
- 5+ window functions implemented: cumulative sum, moving average, ranking, lag/lead
- Multi-partition window specifications (region × department × customer)
- Execution plan analysis with shuffle behavior explanation
- Performance optimization strategies for window operations

**Technical Highlights:**
- Demonstrates `partitionBy()` and `orderBy()` with custom window frames
- Analyzes Catalyst optimizer transformations
- Compares window functions vs traditional GROUP BY + JOIN approaches

### ✅ Task 8: Custom Spark Configuration Tuning
**File:** `task8_configuration_tuning.py`

**Objective:** Systematically test different Spark configurations and analyze their impact on performance.

**Key Features:**
- Benchmarks 4 different configuration sets (Default, Memory-Optimized, Parallelism-Optimized, Aggressive)
- Measures execution time, memory usage, and shuffle behavior
- Analyzes configuration impact on different workloads
- Provides recommendations for optimal settings

**Technical Highlights:**
- Tests `spark.executor.memory`, `spark.sql.shuffle.partitions`, `spark.default.parallelism`
- Includes Adaptive Query Execution (AQE) testing
- Provides practical configuration guidelines

### ✅ Task 6: Persistence Strategy Comparison
**File:** `task6_persistence_strategy.py`

**Objective:** Compare different RDD/DataFrame persistence levels and analyze their impact on iterative tasks.

**Key Features:**
- Compares 3 persistence levels: `MEMORY_ONLY`, `DISK_ONLY`, `MEMORY_AND_DISK`
- Benchmarks iterative computation performance
- Analyzes DAG optimization with caching
- Provides memory management recommendations

**Technical Highlights:**
- Demonstrates correct use of `cache()` and `persist()`
- Shows DAG optimization with cached data
- Analyzes trade-offs between memory usage and performance

## Technical Environment

### Prerequisites
- **Python:** 3.8+
- **Java:** OpenJDK 11 or 17
- **PySpark:** 3.5.0
- **Operating System:** Tested on WSL2 (Ubuntu) and Windows

### Installation
```bash
# Install Java
sudo apt install openjdk-17-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install PySpark
pip install pyspark==3.5.0 or latest 

# Install additional dependencies
pip install pandas numpy
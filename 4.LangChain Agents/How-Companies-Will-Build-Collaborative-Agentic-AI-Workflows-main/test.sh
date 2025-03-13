#!/bin/bash

# Project Proposal Summarizer - Setup Script
# This script creates the directory structure and files for the project

# Display header
echo "======================================================"
echo "  Project Proposal Summarizer - Setup Script"
echo "======================================================"
echo

# Create directory structure
echo "Creating directory structure..."
mkdir -p agents
mkdir -p data
mkdir -p models
mkdir -p processing
mkdir -p tasks
mkdir -p utils
mkdir -p logs

# Create config files
echo "Creating configuration files..."
cat > config.json << 'EOF'
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
  },
  "model": {
    "summarization": {
      "name": "sshleifer/distilbart-cnn-12-6",
      "parameters": {
        "max_length": 150,
        "min_length": 50,
        "do_sample": false
      }
    }
  },
  "agents": {
    "initial_analyzer": {
      "role": "Project Proposal Analyzer",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "market_research": {
      "role": "Market Research Specialist",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "technical_evaluation": {
      "role": "Technical Feasibility Expert",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "financial_analyst": {
      "role": "Financial Viability Assessor",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "risk_assessment": {
      "role": "Risk Management Specialist",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "executive_summarizer": {
      "role": "Executive Summary Specialist",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },
  "process": {
    "type": "sequential",
    "verbose": true
  }
}
EOF

cat > settings.yaml << 'EOF'
# Application Settings
app:
  name: "Project Proposal Summarizer"
  description: "AI-powered analysis and summarization of project proposals"
  version: "1.0.0"

# Logging Configuration
logging:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"
  console:
    enabled: true
    colored: true
  file:
    enabled: false
    path: "logs/proposal_analysis.log"
    rotation: "5 MB"
    backup_count: 5

# Display Settings
display:
  colors:
    header: '\033[95m'
    blue: '\033[94m'
    cyan: '\033[96m'
    green: '\033[92m'
    yellow: '\033[93m'
    red: '\033[91m'
    end: '\033[0m'
    bold: '\033[1m'
    underline: '\033[4m'
  symbols:
    info: 'ℹ'
    debug: '🔍'
    warning: '⚠'
    error: '❌'
    success: '✅'
    highlight: '🔆'

# Process Configuration
process:
  random_delay:
    min: 0.05
    max: 0.3
  summarization:
    model: "sshleifer/distilbart-cnn-12-6"
    max_length: 150
    min_length: 50
    do_sample: false
EOF

cat > constants.js << 'EOF'
/**
 * Application constants for the Project Proposal Summarizer
 */

// Log types
const LOG_TYPES = {
  JSON: 'json',
  TEXT: 'text',
  TABLE: 'table',
  RAW: 'raw',
  ERROR: 'error',
  WARNING: 'warning'
};

// Log levels
const LOG_LEVELS = {
  INFO: 'info',
  DEBUG: 'debug',
  WARNING: 'warning',
  ERROR: 'error',
  SUCCESS: 'success',
  HIGHLIGHT: 'highlight'
};

// ANSI color codes for terminal output
const TERM_COLORS = {
  HEADER: '\033[95m',
  BLUE: '\033[94m',
  CYAN: '\033[96m',
  GREEN: '\033[92m',
  YELLOW: '\033[93m',
  RED: '\033[91m',
  ENDC: '\033[0m',
  BOLD: '\033[1m',
  UNDERLINE: '\033[4m'
};

// Log symbols
const LOG_SYMBOLS = {
  INFO: 'ℹ',
  DEBUG: '🔍',
  WARNING: '⚠',
  ERROR: '❌',
  SUCCESS: '✅',
  HIGHLIGHT: '🔆'
};

// Agent roles
const AGENT_ROLES = {
  INITIAL_ANALYZER: 'Project Proposal Analyzer',
  MARKET_RESEARCH: 'Market Research Specialist',
  TECHNICAL_EVALUATION: 'Technical Feasibility Expert',
  FINANCIAL_ANALYST: 'Financial Viability Assessor',
  RISK_ASSESSMENT: 'Risk Management Specialist',
  EXECUTIVE_SUMMARIZER: 'Executive Summary Specialist'
};

// Default processing delays
const PROCESSING_DELAYS = {
  MIN: 0.05,
  MAX: 0.3
};

// Export all constants
module.exports = {
  LOG_TYPES,
  LOG_LEVELS,
  TERM_COLORS,
  LOG_SYMBOLS,
  AGENT_ROLES,
  PROCESSING_DELAYS
};
EOF

cat > utils.js << 'EOF'
/**
 * Utility functions for the Project Proposal Summarizer
 */

const { LOG_TYPES, LOG_LEVELS, TERM_COLORS, LOG_SYMBOLS } = require('./constants');
const uuid = require('uuid');
const { datetime } = require('node-datetime');
const tabulate = require('tabulate');
const colors = require('colors/safe');

// Global counters for different log types
const logCounters = {
  json: 0,
  text: 0,
  table: 0,
  raw: 0,
  error: 0,
  warning: 0
};

// Generate unique process ID for this run
const PROCESS_ID = uuid.v4();
const SESSION_START = new Date().toISOString();

/**
 * Advanced logging function with multiple output formats and random delays
 * @param {string|object} message - The message or data to log
 * @param {string} level - Log level (info, debug, warning, error, success, highlight)
 * @param {string} logType - Type of log (text, json, table, raw, error)
 */
function logWithTimestamp(message, level = 'info', logType = 'text') {
  // Update counter for this log type
  logCounters[logType in logCounters ? logType : 'text']++;

  const timestamp = new Date().toISOString().replace('T', ' ').substr(0, 23);
  const logId = `${logType[0].toUpperCase()}${String(logCounters[logType]).padStart(5, '0')}`;

  // Basic metadata for all log types
  const metadata = {
    timestamp,
    level,
    logId,
    processId: PROCESS_ID,
    logType,
    sessionUptime: getSessionUptime()
  };

  // Format log based on type
  switch(logType) {
    case LOG_TYPES.JSON:
      logJsonFormat(message, metadata);
      break;
    case LOG_TYPES.TABLE:
      logTableFormat(message);
      break;
    case LOG_TYPES.RAW:
      logRawFormat(logId, timestamp);
      break;
    case LOG_TYPES.ERROR:
      logErrorFormat(message, logId, timestamp);
      break;
    default:
      logTextFormat(message, level, logId, timestamp);
  }
}

module.exports = {
  logWithTimestamp,
  PROCESS_ID,
  SESSION_START
};
EOF

# Create initialization files
echo "Creating __init__.py files..."
touch agents/__init__.py
touch data/__init__.py
touch models/__init__.py
touch processing/__init__.py
touch tasks/__init__.py
touch utils/__init__.py

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
crewai
crewai-tools
transformers
onnxruntime
torch
sentencepiece
protobuf

# Visualization and Output Formatting
tabulate
termcolor
colorama
tqdm
rich

# Data Processing
pandas
numpy
scipy

# Utilities
python-dateutil
pytz
pyyaml
jinja2
humanize

# Optional Performance Monitoring
psutil
memory-profiler
EOF

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# Project specific
logs/
.DS_Store
EOF

# Create README
echo "Creating README.md..."
cat > README.md << 'EOF'
# Project Proposal Summarizer

An AI-powered tool for analyzing and summarizing project proposals using agent-based workflows.

## Overview

The Project Proposal Summarizer is a sophisticated tool that leverages multiple AI agents, each with specialized roles, to analyze project proposals from different perspectives and generate comprehensive, insightful summaries and recommendations.

## Key Features

- **Multi-Agent Analysis**: Uses six specialized agents with distinct roles:
  - Project Proposal Analyzer
  - Market Research Specialist
  - Technical Feasibility Expert
  - Financial Viability Assessor
  - Risk Management Specialist
  - Executive Summary Specialist

- **Comprehensive Evaluation**: Analyzes proposals across multiple dimensions:
  - Business objectives and strategic alignment
  - Market potential and competitive landscape
  - Technical feasibility and architecture
  - Financial projections and ROI
  - Risk assessment and mitigation strategies

- **Advanced Summarization**: Uses transformer-based models to generate concise yet comprehensive executive summaries

- **Detailed Logging**: Provides rich, multi-format logging for transparency and debugging

## Project Structure

```
project-proposal-summarizer/
├── agents/
│   ├── __init__.py
│   └── definitions.py
├── data/
│   ├── __init__.py
│   └── proposal.py
├── models/
│   ├── __init__.py
│   └── summarization.py
├── processing/
│   ├── __init__.py
│   └── simulation.py
├── tasks/
│   ├── __init__.py
│   └── definitions.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── system_info.py
├── config.json
├── settings.yaml
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-proposal-summarizer.git
   cd project-proposal-summarizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python main.py
```

This will:
1. Load the sample project proposal
2. Initialize the agent-based workflow
3. Simulate the analysis process with detailed logging
4. Generate a final summary report

## Configuration

The application can be configured using:

- `config.json`: JSON-based configuration for agents, models, and process settings
- `settings.yaml`: YAML-based configuration for logging, display settings, and process parameters

## Requirements

See `requirements.txt` for a complete list of dependencies. Key requirements include:

- crewai
- transformers
- tabulate
- termcolor
- pandas
- numpy

## License

MIT License
EOF

# Make main.py executable
echo "Creating sample files in each directory..."

# Create empty files to maintain structure
touch agents/definitions.py
touch data/proposal.py
touch models/summarization.py
touch processing/simulation.py
touch tasks/definitions.py
touch utils/config.py
touch utils/logger.py
touch utils/system_info.py

# Make the main script executable
chmod +x main.py

echo "Creating setup.py file..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="project-proposal-summarizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered tool for analyzing and summarizing project proposals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/project-proposal-summarizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "proposal-summarizer=main:main",
        ],
    },
)
EOF

cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
Project Proposal Summarizer - Main Entry Point
"""

import logging
import time
import platform
import sys
import os
from crewai import Crew, Process

# Import project modules
from utils.logger import log_with_timestamp
from utils.system_info import get_system_info, print_system_info
from agents.definitions import create_agents
from tasks.definitions import create_tasks
from processing.simulation import simulate_agent_processing
from models.summarization import summarize_text, format_final_report
from data.proposal import get_sample_proposal


def main():
    """Main entry point for the Project Proposal Summarizer"""
    # Print system information
    print_system_info()

    # Log application start
    log_with_timestamp("Project Proposal Summarizer starting...", "info")

    # Get sample proposal text
    log_with_timestamp("Preparing example project proposal for analysis...", "info")
    project_proposal_text = get_sample_proposal()
    log_with_timestamp(f"Project proposal loaded - {len(project_proposal_text)} characters", "info")

    # Create agents
    (
        initial_analyzer_agent,
        market_research_agent,
        technical_evaluation_agent,
        financial_analyst_agent,
        risk_assessment_agent,
        executive_summarizer_agent
    ) = create_agents()

    # Create tasks
    (
        initial_analysis_task,
        market_research_task,
        technical_evaluation_task,
        financial_analysis_task,
        risk_assessment_task,
        executive_summary_task
    ) = create_tasks(
        initial_analyzer_agent,
        market_research_agent,
        technical_evaluation_agent,
        financial_analyst_agent,
        risk_assessment_agent,
        executive_summarizer_agent
    )

    # Configure the crew
    log_with_timestamp("Configuring crew with sequential processing workflow...", "info")
    crew = Crew(
        agents=[
            initial_analyzer_agent,
            market_research_agent,
            technical_evaluation_agent,
            financial_analyst_agent,
            risk_assessment_agent,
            executive_summarizer_agent
        ],
        tasks=[
            initial_analysis_task,
            market_research_task,
            technical_evaluation_task,
            financial_analysis_task,
            risk_assessment_task,
            executive_summary_task
        ],
        process=Process.sequential,
        verbose=True
    )

    log_with_timestamp("Crew configuration complete! All agents and tasks successfully initialized.", "info")

    # Execute the workflow (using simulation)
    log_with_timestamp("Beginning sequential agent workflow execution", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Initial Analyzer Agent
    initial_analysis_output = simulate_agent_processing("Project Proposal Analyzer", project_proposal_text)
    log_with_timestamp(f"Initial Analysis Output: {initial_analysis_output}", "info")
    log_with_timestamp("Transferring initial analysis to Market Research Agent...", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Market Research Agent
    market_analysis_output = simulate_agent_processing("Market Research Specialist", initial_analysis_output)
    log_with_timestamp(f"Market Analysis Output: {market_analysis_output}", "info")
    log_with_timestamp("Transferring market analysis to Technical Evaluation Agent...", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Technical Evaluation Agent
    technical_evaluation_output = simulate_agent_processing("Technical Feasibility Expert", market_analysis_output)
    log_with_timestamp(f"Technical Evaluation Output: {technical_evaluation_output}", "info")
    log_with_timestamp("Transferring technical evaluation to Financial Analyst Agent...", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Financial Analyst Agent
    financial_analysis_output = simulate_agent_processing("Financial Viability Assessor", technical_evaluation_output)
    log_with_timestamp(f"Financial Analysis Output: {financial_analysis_output}", "info")
    log_with_timestamp("Transferring financial analysis to Risk Assessment Agent...", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Risk Assessment Agent
    risk_assessment_output = simulate_agent_processing("Risk Management Specialist", financial_analysis_output)
    log_with_timestamp(f"Risk Assessment Output: {risk_assessment_output}", "info")
    log_with_timestamp("Transferring risk assessment to Executive Summarizer Agent...", "info")
    log_with_timestamp("============================================================", "info")

    # Simulate Executive Summarizer Agent
    executive_summary_output = simulate_agent_processing("Executive Summary Specialist", risk_assessment_output)
    log_with_timestamp(f"Executive Summary Output: {executive_summary_output}", "info")
    log_with_timestamp("============================================================", "info")

    log_with_timestamp("All agents have completed their tasks. Generating final output...", "info")

    # Combine all outputs for final summarization
    combined_analysis = f"""
# NexGen Enterprise Analytics Platform (NEAP) Analysis

## Initial Analysis:
{initial_analysis_output}

## Market Research:
{market_analysis_output}

## Technical Evaluation:
{technical_evaluation_output}

## Financial Analysis:
{financial_analysis_output}

## Risk Assessment:
{risk_assessment_output}

## Executive Summary:
{executive_summary_output}
"""

    log_with_timestamp(f"Combined analysis document created: {len(combined_analysis)} characters", "info")

    # Generate summary
    summary_text = summarize_text(combined_analysis)

    # Format final report
    final_report = format_final_report(summary_text)

    # Print the final output
    print("\n\n" + "=" * 80)
    print("\nFINAL PROJECT PROPOSAL ANALYSIS:\n")
    print(final_report)
    print("\n" + "=" * 80)

    log_with_timestamp("Project proposal analysis workflow completed successfully!", "info")


if __name__ == "__main__":
    main()
EOF
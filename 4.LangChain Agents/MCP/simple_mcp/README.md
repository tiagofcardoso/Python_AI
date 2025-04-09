# MCP Server - Financial Data Analysis

## Overview

The MCP Server (Model-Controller-Processor) is a system that integrates a large language model (LLM) with a SQL Server database for financial data analysis. The system allows users to make natural language queries about financial market data, which are then interpreted by the LLM and converted into SQL queries or specific analyses.

## Architecture

The system consists of three main components:

1. **SQLServerConnection**: Manages the connection to the SQL Server database
2. **DatabaseAgent**: Executes queries and analyses on the database
3. **MCPServer**: Coordinates the interaction between the user, the LLM, and the database

### Execution Flow

1. The user enters a query in natural language
2. The MCPServer sends the query to the LLM for interpretation
3. The LLM determines if the query requires database access
4. If necessary, the DatabaseAgent executes the SQL query or stock analysis
5. The results are formatted and presented to the user

## Main Features

### Direct SQL Queries
The system can translate natural language questions into SQL queries, allowing users without technical knowledge to access database data.

### Stock Analysis
The system can analyze historical stock data and provide important metrics such as:
- Average price
- Standard deviation
- Maximum and minimum prices
- Average volume
- Recommendations based on volatility

### Informative Responses
For queries that don't require database data, the LLM provides direct responses based on its knowledge of finance and markets.

### Error Handling
The system handles exceptions and provides clear error messages to facilitate debugging and improve user experience.

## Detailed Components

### SQLServerConnection
This class manages the connection to the SQL Server database:
- Configures a connection string for the local SQL Server on port 1433
- Uses the "MercadoFinanceiro" database
- Uses "SA" user credentials and password from environment or default
- The `connect()` method returns an active database connection

### DatabaseAgent
This class executes operations on the database:

#### query_database Method
- Executes a SQL query provided as a parameter
- Handles different types of queries (SELECT vs. INSERT/UPDATE/DELETE)
- Converts the results into a list of dictionaries for easy manipulation
- Handles exceptions and returns appropriate error messages

#### analyze_stock Method
- Queries historical data for a specific stock over a determined period
- Converts the results into a pandas DataFrame for analysis
- Calculates important metrics such as average price, standard deviation, maximums and minimums
- Evaluates the stock's volatility and generates a recommendation based on this analysis
- Returns a structured report with all relevant information

### MCPServer
This class coordinates the interaction between the user, the LLM, and the database:

#### process_query Method
- Receives a natural language query from the user
- Uses the LLM to interpret the user's intention
- Determines if the query requires database access
- Directs the query to the appropriate DatabaseAgent method
- Returns appropriately formatted results

#### _query_llm Method
- Sends the user's query to the GPT-4o model
- Provides a detailed system prompt that instructs the LLM on how to interpret queries
- Requests response in JSON format to facilitate processing
- Handles exceptions and returns appropriate error messages

## Requirements

- Python 3.8+
- pyodbc
- pandas
- openai
- SQL Server with ODBC driver
- "MercadoFinanceiro" database configured with appropriate tables

## Configuration

1. Set the `OPENAI_API_KEY` environment variable with your OpenAI API key
2. Set the `SQL_PASSWORD` environment variable with your SQL Server password
3. Make sure SQL Server is running on port 1433
4. Verify that the "MercadoFinanceiro" database exists with the necessary tables

## Usage

Run the main script to start the command-line interface:

```bash
python finacial_analyser.py
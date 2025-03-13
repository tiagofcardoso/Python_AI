"""
Simulation utilities for agent processing
"""

import random
import time
import uuid
from datetime import datetime

from utils.logger import log_with_timestamp


def simulate_agent_processing(agent_name, content):
    """
    Simulate ultra-detailed processing logs for each agent with varied log formats

    Args:
        agent_name (str): Name of the agent
        content (str): Input content for the agent

    Returns:
        str: Simulated agent output summary
    """
    agent_id = f"agent-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4().hex[:16]}"

    # Start with a JSON log for agent initialization
    init_log = {
        'event': 'agent_initialization',
        'agent_name': agent_name,
        'agent_id': agent_id,
        'trace_id': trace_id,
        'input_size': len(content),
        'timestamp_utc': datetime.utcnow().isoformat(),
        'configuration': {
            'max_tokens': random.randint(2048, 8192),
            'temperature': round(random.uniform(0.1, 0.8), 2),
            'top_p': round(random.uniform(0.9, 1.0), 2),
            'frequency_penalty': round(random.uniform(0.0, 0.5), 2),
            'presence_penalty': round(random.uniform(0.0, 0.5), 2),
            'analysis_depth': random.choice(['shallow', 'medium', 'deep', 'exhaustive']),
            'context_window': random.randint(4096, 16384),
            'knowledge_cutoff': random.choice(['2022-01', '2022-06', '2022-12', '2023-04', '2023-09'])
        },
        'system_metrics': {
            'memory_available_mb': random.randint(1024, 16384),
            'node_id': f"node-{random.randint(1, 20):02d}",
            'cluster': random.choice(['us-east', 'us-west', 'eu-central']),
            'gpu_utilization': round(random.uniform(0.2, 0.8), 2)
        }
    }
    log_with_timestamp(init_log, 'info', 'json')

    # Log agent loading with table format
    memory_stats = [
        ['Knowledge Graph', f"{random.randint(100, 500):,} nodes", f"{random.randint(500, 2000):,} edges",
         f"{random.randint(50, 200):.2f} MB"],
        ['Domain Corpus', f"{random.randint(10000, 50000):,} documents", f"{random.randint(1, 50):,} million tokens",
         f"{random.randint(200, 1000):.2f} MB"],
        ['Task Templates', f"{random.randint(50, 200):,} templates", f"{random.randint(5, 30):,} categories",
         f"{random.randint(10, 50):.2f} MB"],
        ['Reasoning Modules', f"{random.randint(10, 50):,} modules", f"{random.randint(50, 200):,} functions",
         f"{random.randint(20, 100):.2f} MB"],
        ['Agent Memory', f"{random.randint(1000, 5000):,} entries", f"{random.randint(10, 100):,} sessions",
         f"{random.randint(50, 200):.2f} MB"]
    ]

    log_with_timestamp({
        'title': f"AGENT MEMORY ALLOCATION TABLE - {agent_name}",
        'headers': ['Component', 'Size', 'Elements', 'Memory Usage'],
        'table_data': memory_stats,
        'format': 'fancy_grid',
        'footer': f"Total Memory Allocation: {sum([float(row[3].split()[0]) for row in memory_stats]):.2f} MB | Agent ID: {agent_id}"
    }, 'info', 'table')

    # Show raw memory-like logs
    log_with_timestamp(f"Loading neural weights for {agent_name}", 'debug', 'raw')

    # Main processing steps with mixed log formats
    steps = [
        "Initializing task parameters and context loading",
        "Parsing input document and tokenizing content stream",
        "Extracting structured information and semantic entities",
        "Building knowledge representation from input data",
        "Identifying key entities and relationship networks",
        "Applying domain-specific analysis frameworks",
        "Generating preliminary insights based on content analysis",
        "Running statistical validation on initial findings",
        "Cross-referencing findings with agent knowledge base",
        "Validating conclusions against factual constraints",
        "Performing counterfactual reasoning on key assertions",
        "Generating confidence intervals for numerical predictions",
        "Formulating structured response according to task requirements",
        "Applying quality checks to ensure output completeness",
        "Optimizing response format for downstream consumption",
        "Finalizing response formatting and metadata enrichment"
    ]

    # Track some mock metrics through the processing steps
    confidence_trend = []
    memory_usage_trend = []
    processing_time_trend = []
    token_usage = 0

    for i, step in enumerate(steps):
        # Choose a random log type for variety
        log_type = random.choice(['text', 'text', 'text', 'json', 'table', 'raw', 'error'])

        # Update mock metrics
        current_confidence = round(random.uniform(0.7 + (i * 0.01), 0.99), 3)
        confidence_trend.append(current_confidence)
        current_memory = random.randint(200 + (i * 10), 400 + (i * 20))
        memory_usage_trend.append(current_memory)
        current_processing = random.randint(50, 500)
        processing_time_trend.append(current_processing)
        token_usage += random.randint(50, 200)

        # Generate different log types
        if log_type == 'text':
            _simulation_log_text(step, i, len(steps), agent_name)
        elif log_type == 'json':
            _simulation_log_json(step, i, agent_id, trace_id, current_confidence,
                                 current_memory, current_processing, token_usage)
        elif log_type == 'table' and i > 2:
            _simulation_log_table(step, i, confidence_trend, memory_usage_trend,
                                  processing_time_trend, agent_id)
        elif log_type == 'raw' and random.random() < 0.5:
            log_with_timestamp(f"Memory dump during step {i + 1}: {step}", 'debug', 'raw')
        elif log_type == 'error' and random.random() < 0.2:
            _simulation_log_error(step, i)

        time.sleep(random.uniform(0.2, 0.5))  # Simulate processing time

        # Generate some random processing details with varied log types
        _simulation_log_details(i, step)

        # Occasionally log a warning or issue with JSON format
        _simulation_log_warnings(i)

    # Show final metrics table
    _simulation_log_final_metrics(agent_name, confidence_trend, memory_usage_trend,
                                  processing_time_trend)

    # Final JSON performance log
    _simulation_log_performance(agent_id, agent_name, trace_id, confidence_trend,
                                memory_usage_trend, token_usage, steps, processing_time_trend)

    # Log completion with success message
    log_with_timestamp(f"Agent '{agent_name}' completed task successfully", 'success')

    # Return a detailed output section based on the agent
    agent_output = _get_agent_output(agent_name)

    # Log the structured output as a JSON entry
    output_log = {
        'event': 'agent_output_generated',
        'agent_id': agent_id,
        'agent_name': agent_name,
        'trace_id': trace_id,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'output': agent_output
    }
    log_with_timestamp(output_log, 'info', 'json')

    return agent_output["summary"]


def _simulation_log_text(step, i, total_steps, agent_name):
    """Log text format for simulation"""
    log_level = random.choice(['info', 'debug', 'info', 'info', 'success'])
    log_with_timestamp(f"Agent '{agent_name}' - Step {i + 1}/{total_steps}: {step}", log_level)


def _simulation_log_json(step, i, agent_id, trace_id, confidence, memory, processing_time, token_usage):
    """Log JSON format for simulation"""
    json_log = {
        'event': 'processing_step',
        'agent_id': agent_id,
        'trace_id': trace_id,
        'step_id': i + 1,
        'step_name': step,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'metrics': {
            'confidence': confidence,
            'memory_usage_mb': memory,
            'processing_time_ms': processing_time,
            'token_usage': token_usage,
            'entropy': round(random.uniform(0.1, 0.9), 3)
        },
        'step_details': {
            'inputs_processed': random.randint(1, 5),
            'operations_executed': random.randint(5, 20),
            'cache_hit_ratio': round(random.uniform(0.5, 0.95), 2),
            'parallel_threads': random.randint(1, 8)
        }
    }
    log_with_timestamp(json_log, 'info', 'json')


def _simulation_log_table(step, i, confidence_trend, memory_usage_trend, processing_time_trend, agent_id):
    """Log table format for simulation"""
    metrics_table = [
        ['Confidence', f"{confidence_trend[-1]:.3f}", f"{sum(confidence_trend) / len(confidence_trend):.3f}",
         f"{min(confidence_trend):.3f}", f"{max(confidence_trend):.3f}"],
        ['Memory (MB)', f"{memory_usage_trend[-1]}", f"{sum(memory_usage_trend) / len(memory_usage_trend):.1f}",
         f"{min(memory_usage_trend)}", f"{max(memory_usage_trend)}"],
        ['Processing (ms)', f"{processing_time_trend[-1]}",
         f"{sum(processing_time_trend) / len(processing_time_trend):.1f}", f"{min(processing_time_trend)}",
         f"{max(processing_time_trend)}"]
    ]

    log_with_timestamp({
        'title': f"PROCESSING METRICS - Step {i + 1}: {step}",
        'headers': ['Metric', 'Current', 'Average', 'Min', 'Max'],
        'table_data': metrics_table,
        'format': 'grid',
        'footer': f"Process ID: {agent_id} | Elapsed Steps: {i + 1}/{len(steps)}"
    }, 'info', 'table')


def _simulation_log_error(step, i):
    """Log error format for simulation"""
    log_with_timestamp(f"Recoverable issue detected during {step}", 'warning', '
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
    market_analysis_output = simulate_
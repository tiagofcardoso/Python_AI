"""
Task definitions for the Project Proposal Summarizer
"""

from crewai import Task

from utils.logger import log_with_timestamp


def create_tasks(
        initial_analyzer_agent,
        market_research_agent,
        technical_evaluation_agent,
        financial_analyst_agent,
        risk_assessment_agent,
        executive_summarizer_agent
):
    """
    Create and return all tasks with detailed descriptions and expected outputs

    Args:
        initial_analyzer_agent: Agent for initial analysis
        market_research_agent: Agent for market research
        technical_evaluation_agent: Agent for technical evaluation
        financial_analyst_agent: Agent for financial analysis
        risk_assessment_agent: Agent for risk assessment
        executive_summarizer_agent: Agent for executive summary

    Returns:
        tuple: A tuple containing all six task instances
    """
    log_with_timestamp("Configuring agent-specific tasks with comprehensive acceptance criteria...", "info")

    initial_analysis_task = Task(
        description="""Perform a comprehensive initial analysis of the project proposal document.

        Your analysis must include:
        1. Identification of the core business problem the proposal aims to solve
        2. Clear articulation of the proposed solution and its key components
        3. Extraction of stated business objectives and success metrics
        4. Identification of key stakeholders and their interests
        5. Analysis of the proposal's alignment with organizational strategy
        6. Preliminary identification of potential strengths and weaknesses
        7. Highlighting of any missing critical information

        Flag any instances of vague language, unsupported claims, or logical inconsistencies.
        Apply specific business analysis frameworks where appropriate (SWOT, PESTEL, etc.).
        Identify any unstated assumptions that might impact project success.
        """,
        expected_output="""A structured, detailed analysis document with clearly labeled sections addressing each required component.
        The analysis should be 1-2 pages in length, use bullet points for clarity, and include direct quotes from the proposal to support your observations.
        Include a "Critical Questions" section highlighting areas requiring further clarification or investigation.
        """,
        agent=initial_analyzer_agent
    )

    market_research_task = Task(
        description="""Conduct a thorough market analysis based on the project proposal and initial analysis.

        Your analysis must include:
        1. Evaluation of the target market size, growth rate, and segmentation
        2. Assessment of customer pain points and how well the proposal addresses them
        3. Competitive landscape analysis, including direct and indirect competitors
        4. Identification of market entry barriers and how the proposal plans to overcome them
        5. Analysis of market trends and how they might impact project success
        6. Evaluation of the proposal's unique value proposition and competitive advantages
        7. Assessment of pricing strategy and revenue model viability

        Base your analysis on factual market data wherever possible. Critically examine any market claims made in the proposal.
        Identify any market risks or opportunities that the proposal may have overlooked.
        """,
        expected_output="""A comprehensive market analysis document with sections addressing each required component.
        Include visual elements such as competitor positioning maps and market sizing charts where appropriate.
        Provide a clear "Market Verdict" section stating whether the proposal's market assumptions are realistic.
        Flag any market claims that appear exaggerated or unsupported by data.
        """,
        agent=market_research_agent,
        context=[initial_analysis_task]  # This task depends on the initial analysis
    )

    technical_evaluation_task = Task(
        description="""Evaluate the technical feasibility and appropriateness of the proposed solution.

        Your evaluation must include:
        1. Assessment of the proposed technology stack and architecture
        2. Evaluation of the development approach and methodology
        3. Analysis of infrastructure requirements and scalability considerations
        4. Identification of potential technical dependencies and integration points
        5. Assessment of technical resource requirements and availability
        6. Evaluation of the technical implementation timeline and milestones
        7. Identification of potential technical risks and mitigation strategies
        8. Analysis of security and compliance considerations

        For each technical component, provide a feasibility rating (High/Medium/Low) with justification.
        Identify any technical debt that might be incurred and its long-term implications.
        """,
        expected_output="""A detailed technical evaluation report with sections addressing each required component.
        Include a technical architecture diagram if possible. 
        Provide a clear "Technical Feasibility Verdict" with an overall rating and key considerations.
        Include specific recommendations for addressing any identified technical issues or risks.
        """,
        agent=technical_evaluation_agent,
        context=[initial_analysis_task]  # This task depends on the initial analysis
    )

    financial_analysis_task = Task(
        description="""Analyze the financial aspects of the project proposal with a focus on viability and return on investment.

        Your analysis must include:
        1. Detailed review of cost projections, including CAPEX and OPEX
        2. Evaluation of revenue forecasts and underlying assumptions
        3. Calculation of key financial metrics (ROI, NPV, IRR, payback period)
        4. Assessment of funding requirements and potential sources
        5. Analysis of cash flow projections and working capital needs
        6. Stress testing of financial projections under different scenarios
        7. Identification of financial risks and sensitivities

        Challenge any overly optimistic assumptions. Apply industry benchmarks where appropriate.
        Identify any hidden costs or revenue dependencies that might impact financial outcomes.
        """,
        expected_output="""A comprehensive financial analysis with clearly structured sections for costs, revenues, and financial metrics.
        Include financial tables and charts illustrating key points.
        Provide a "Financial Viability Verdict" with clear recommendations.
        Include a sensitivity analysis showing how changes in key variables would impact financial outcomes.
        """,
        agent=financial_analyst_agent,
        context=[initial_analysis_task, market_research_task]  # This task depends on prior analyses
    )

    risk_assessment_task = Task(
        description="""Conduct a thorough risk assessment of the project proposal, covering all potential risk categories.

        Your assessment must include:
        1. Identification and rating of market risks (demand uncertainty, competitive threats, etc.)
        2. Analysis of technical risks (technology maturity, integration challenges, etc.)
        3. Evaluation of operational risks (resource availability, dependency management, etc.)
        4. Assessment of financial risks (cost overruns, revenue shortfalls, funding issues, etc.)
        5. Identification of regulatory and compliance risks
        6. Analysis of reputational and strategic risks
        7. Evaluation of the proposal's risk management approach and contingency plans

        For each identified risk, provide a probability rating (High/Medium/Low), potential impact rating, and suggested mitigation strategies.
        Identify any "show-stopper" risks that could fundamentally undermine project success.
        """,
        expected_output="""A detailed risk register with categorized risks, probability and impact ratings, and mitigation strategies.
        Include a risk heat map visualizing the distribution of risks by probability and impact.
        Provide a "Risk Assessment Verdict" summarizing the overall risk profile of the proposal.
        Identify the top 5 risks that should receive immediate attention if the project proceeds.
        """,
        agent=risk_assessment_agent,
        context=[initial_analysis_task, market_research_task, technical_evaluation_task, financial_analysis_task]
        # This task depends on all prior analyses
    )

    executive_summary_task = Task(
        description="""Create a comprehensive yet concise executive summary of the project proposal and all associated analyses.

        Your executive summary must include:
        1. Brief overview of the proposal's core business problem and proposed solution
        2. Summary of key findings from each analysis domain (market, technical, financial, risk)
        3. Highlights of the proposal's strengths and competitive advantages
        4. Honest assessment of key concerns, risks, or limitations
        5. Clear recommendations on whether to proceed, modify, or reject the proposal
        6. If proceeding, outline of suggested next steps and critical success factors
        7. If modifications are needed, prioritized list of changes required

        Strike a balance between positive and critical perspectives. Be honest but constructive.
        Ensure the summary is accessible to both technical and non-technical executives.
        """,
        expected_output="""A 2-page executive summary with clearly defined sections addressing each required component.
        Begin with a one-paragraph "TL;DR" that states the bottom-line recommendation.
        Use visual elements sparingly but effectively to highlight key points.
        Conclude with a clear "Decision Guidance" section that helps executives make an informed decision.
        """,
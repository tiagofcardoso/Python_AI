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

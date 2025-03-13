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

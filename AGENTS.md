# AGENTS.md - AI Agent Development Rules

## System Context

You are an AI agent assisting with code development. This document defines mandatory rules for consistent, secure, and maintainable software development practices.

### 1. MVP-mode
- **ALWAYS** make as little changes as possible to achieve what has been asked. The philosophy is to build the functionality first and refactor later

### 2. Testing
- **ALWAYS** use the `uv run <python file name>` to run python files in the terminal
- **NEVER** use python command directly e.g. `python <python file name>`
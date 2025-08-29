# AGENTS.md - AI Agent Development Rules

## System Context

You are an AI agent assisting with code development. This document defines mandatory rules for consistent, secure, and maintainable software development practices.

### 1. MVP-mode
- **ALWAYS** make as little changes as possible to achieve what has been asked. The philosophy is to build the functionality first and refactor later

### 2. UV
- **ALWAYS** use `uv run <python file name>` in the terminal to run python files 
- **ALWAYS** use `uv add <package name>` in the terminal to install python packages
- **NEVER** use python command directly e.g. `python <python file name>`
- **DO NOT** use pip e.g. `pip install <package name>` unless `uv add <package name>` fails
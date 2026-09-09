# Text-to-SQL Agent

An LLM-powered automation agent that translates plain-English questions into
SQL, runs them against a real database, and returns human-readable answers —
the same pattern behind "chat with your data" tools used in production BI
and analytics products.

## What it does

Give the agent a natural-language question like *"What is the average salary
in Sales?"* and it will:

1. Introspect the database schema (departments, employees, projects tables
   in an in-memory SQLite database).
2. Generate a SQL query that answers the question — using the real
   Anthropic API if `ANTHROPIC_API_KEY` is set, or a deterministic
   rule-based NL→SQL translator otherwise, so the demo always runs with
   zero cost and zero setup.
3. Validate the generated SQL is a single, read-only `SELECT` statement
   (blocking `INSERT`/`UPDATE`/`DELETE`/`DROP`/etc.) before ever executing
   it — a critical safety gate for any agent that turns free text into
   database queries.
4. Execute the query and print a formatted table of results.

## Why it's interesting

Text-to-SQL is one of the most widely deployed "AI agent" patterns today:
it turns a database only analysts can query into something anyone in a
company can ask questions of directly. This project builds the full
pipeline end-to-end — schema introspection, generation, a safety
validation layer, and execution — rather than just a single prompt-and-hope
call to an LLM.

## Tech stack and key concepts

- **SQLite (`sqlite3`, standard library)** — in-memory sample database with
  three related tables (departments, employees, projects) and foreign keys.
- **Rule-based NL→SQL translator** — a transparent, dependency-free stand-in
  for an LLM backend, handling counts, averages, top-N, threshold filters,
  status filters, and department scoping via keyword/regex matching.
- **Optional real LLM backend** — if `ANTHROPIC_API_KEY` is set and the
  `anthropic` package is installed, the agent calls Claude directly to
  generate SQL from the schema + question instead of using the fallback.
- **SQL safety validation** — regex/keyword-based guardrail that rejects
  anything that isn't a single read-only `SELECT`.

## Installation

```bash
pip install -r requirements.txt
```

The `anthropic` package is only needed if you want to use the real LLM
backend. The script runs with no dependencies at all if you skip this step
and rely on the rule-based fallback.

## How to run it

Run the built-in demo (6 example questions):

```bash
python text_to_sql_agent.py
```

Or ask your own questions as command-line arguments:

```bash
python text_to_sql_agent.py "How many employees are in Marketing?" "Show all completed projects"
```

To use the real Claude API instead of the rule-based fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python text_to_sql_agent.py
```

## Example output

```
========================================================================
TEXT-TO-SQL AGENT DEMO
========================================================================
Backend: rule-based fallback (no API key set)

Q: How many employees are in the Engineering department?
  Generated SQL [rule-based-fallback]:
    SELECT COUNT(*) AS employee_count FROM employees WHERE employees.department_id = (SELECT id FROM departments WHERE lower(name) = 'engineering')
  Result:
    employee_count
    --------------
    4

Q: What is the average salary in Sales?
  Generated SQL [rule-based-fallback]:
    SELECT AVG(salary) AS average_salary FROM employees WHERE employees.department_id = (SELECT id FROM departments WHERE lower(name) = 'sales')
  Result:
    average_salary
    --------------
    125000.0

Q: Which department has the highest budget?
  Generated SQL [rule-based-fallback]:
    SELECT name, budget FROM departments ORDER BY budget DESC LIMIT 1
  Result:
    name        | budget
    ------------+--------
    Engineering | 2500000

Q: Show me the top 3 highest paid employees
  Generated SQL [rule-based-fallback]:
    SELECT name, role, salary FROM employees ORDER BY salary DESC LIMIT 3
  Result:
    name         | role                | salary
    -------------+---------------------+-------
    Omar Haddad  | Staff Engineer      | 172000
    Priya Rao    | Engineering Manager | 168000
    Sofia Torres | Sales Director      | 155000

Q: Show all active projects in Engineering
  Generated SQL [rule-based-fallback]:
    SELECT name, status, budget FROM projects WHERE status = 'active' AND projects.department_id = (SELECT id FROM departments WHERE lower(name) = 'engineering')
  Result:
    name             | status | budget
    -----------------+--------+-------
    Platform Rewrite | active | 400000
    Mobile App v2    | active | 250000
```

## How it works

1. **`build_database()`** creates an in-memory SQLite database and seeds it
   with three related tables: `departments`, `employees`, and `projects`.
2. **`get_schema_description()`** introspects `sqlite_master` to produce a
   text description of the schema — the same context a real LLM-backed
   agent would feed into its prompt.
3. **`TextToSQLAgent.ask()`** is the core loop: it tries `llm_generate_sql()`
   first (only active if `ANTHROPIC_API_KEY` is set), falls back to
   `rule_based_generate_sql()` otherwise, then passes whatever SQL comes
   back through `validate_sql()` before executing it with `sqlite3`.
4. **`rule_based_generate_sql()`** looks for question patterns — aggregate
   verbs (`count`, `average`), superlatives (`highest`, `top N`),
   comparisons (`above`, `below`), and entity/department names — and maps
   them to a parameterized SQL template. It's a simplified, inspectable
   stand-in for what an LLM does implicitly.
5. **`validate_sql()`** is the safety layer: it rejects anything that isn't
   a single statement starting with `SELECT`, and blocks a list of
   destructive keywords (`insert`, `update`, `delete`, `drop`, `alter`,
   `pragma`, etc.) even if they appear disguised inside the query text.
6. **`format_table()`** renders the result set as an aligned plain-text
   table for readability in the terminal.

## Extending it

- Point `build_database()` at a real SQLite file (or swap in
  `psycopg2`/`pyodbc` for Postgres/SQL Server) to query real data.
- Expand `rule_based_generate_sql()` with more patterns, or rely entirely on
  the LLM backend for open-ended questions it wasn't explicitly coded to
  handle.
- Add a conversation loop that lets the agent ask a clarifying question when
  it can't confidently map the question to SQL.

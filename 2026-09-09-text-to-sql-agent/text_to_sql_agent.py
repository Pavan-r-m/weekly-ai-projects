"""
Text-to-SQL Agent
==================

An LLM-powered automation agent that translates plain-English questions into
SQL queries, runs them against a real (in-memory) SQLite database, and
returns human-readable answers.

Why this is interesting
------------------------
Text-to-SQL is one of the most practical "AI agent" patterns in production
today: instead of teaching every stakeholder SQL, you let an LLM read the
database schema and translate a natural-language question into a safe,
read-only query. This project implements the full pipeline:

    natural language question
        -> schema-aware prompt construction
        -> SQL generation (LLM if a key is available, else a deterministic
           rule-based NL->SQL translator used as an offline-friendly stand-in)
        -> SQL safety validation (SELECT-only, no destructive statements)
        -> execution against SQLite
        -> formatted, human-readable answer

No API key is required to run this project. If ANTHROPIC_API_KEY is set in
the environment, the agent will call the real Claude API to generate SQL.
Otherwise it automatically falls back to a built-in rule-based translator
that still produces correct SQL for a useful range of question patterns, so
the whole demo runs end-to-end with zero external dependencies or cost.
"""

import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Sample database: a small "company" schema with departments, employees,
#    and projects. This stands in for "the user's real database" that a
#    text-to-SQL agent would normally be pointed at.
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget INTEGER NOT NULL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_year INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    budget INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
"""

SEED_SQL = """
INSERT INTO departments (id, name, budget) VALUES
    (1, 'Engineering', 2500000),
    (2, 'Sales', 1200000),
    (3, 'Marketing', 800000),
    (4, 'Support', 600000);

INSERT INTO employees (id, name, department_id, role, salary, hire_year) VALUES
    (1, 'Ava Chen', 1, 'Senior Engineer', 145000, 2019),
    (2, 'Marcus Ibe', 1, 'Engineer', 112000, 2021),
    (3, 'Priya Rao', 1, 'Engineering Manager', 168000, 2017),
    (4, 'Liam OConnor', 2, 'Account Executive', 95000, 2020),
    (5, 'Sofia Torres', 2, 'Sales Director', 155000, 2016),
    (6, 'Ken Watanabe', 3, 'Marketing Lead', 98000, 2018),
    (7, 'Nadia Petrova', 3, 'Content Strategist', 78000, 2022),
    (8, 'Diego Alvarez', 4, 'Support Engineer', 82000, 2021),
    (9, 'Grace Kim', 4, 'Support Manager', 105000, 2019),
    (10, 'Omar Haddad', 1, 'Staff Engineer', 172000, 2015);

INSERT INTO projects (id, name, department_id, status, budget) VALUES
    (1, 'Platform Rewrite', 1, 'active', 400000),
    (2, 'Mobile App v2', 1, 'active', 250000),
    (3, 'Legacy Cleanup', 1, 'completed', 90000),
    (4, 'Q3 Outbound Push', 2, 'active', 60000),
    (5, 'Enterprise Expansion', 2, 'active', 150000),
    (6, 'Brand Refresh', 3, 'completed', 45000),
    (7, 'Help Center Overhaul', 4, 'active', 30000);
"""


def build_database() -> sqlite3.Connection:
    """Create and seed the in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_SQL)
    conn.commit()
    return conn


def get_schema_description(conn: sqlite3.Connection) -> str:
    """Introspect the database and produce a schema description for the
    prompt/translator, the same way a real text-to-SQL agent would."""
    cursor = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table'"
    )
    lines = []
    for name, create_sql in cursor.fetchall():
        lines.append(f"Table {name}:\n{create_sql}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. SQL safety validation. Whatever generates the SQL (LLM or fallback),
#    we never trust it blindly -- a real agent must guard against
#    destructive or out-of-scope statements before execution.
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "pragma", "replace", "truncate",
)


class UnsafeSQLError(Exception):
    """Raised when generated SQL fails the safety check."""


def validate_sql(sql: str) -> str:
    """Ensure the generated SQL is a single, read-only SELECT statement."""
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()

    if not lowered.startswith("select"):
        raise UnsafeSQLError(f"Only SELECT statements are allowed. Got: {sql!r}")

    if ";" in cleaned:
        raise UnsafeSQLError("Multiple statements are not allowed.")

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise UnsafeSQLError(f"Forbidden keyword '{keyword}' detected in SQL.")

    return cleaned


# ---------------------------------------------------------------------------
# 3. SQL generation. Two backends:
#      (a) real LLM call, used only if ANTHROPIC_API_KEY is set
#      (b) a deterministic, rule-based NL->SQL translator used as a free,
#          offline-friendly fallback so this project runs for anyone.
# ---------------------------------------------------------------------------

DEPARTMENTS = ["engineering", "sales", "marketing", "support"]


def llm_generate_sql(question: str, schema: str) -> Optional[str]:
    """Attempt to generate SQL using the real Anthropic API. Returns None if
    no API key is configured or the SDK is not installed, so callers can
    fall back gracefully."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        print("  (anthropic package not installed; using offline fallback)")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are a SQL generation agent. Given the SQLite schema below and a "
        "natural-language question, respond with ONLY a single read-only "
        "SELECT statement that answers the question. No explanation, no "
        "markdown fences.\n\n"
        f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    )
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return sql.strip()


def rule_based_generate_sql(question: str) -> str:
    """A small, deterministic NL->SQL translator covering common analytic
    question patterns: counts, averages, filters, top-N, and status filters.
    This is intentionally readable "glass box" logic (as opposed to an LLM's
    black box) so the pipeline is fully inspectable and runs with zero
    external dependencies.
    """
    q = question.lower()

    department = next((d for d in DEPARTMENTS if d in q), None)
    dept_filter = ""
    if department:
        dept_filter = (
            f"employees.department_id = "
            f"(SELECT id FROM departments WHERE lower(name) = '{department}')"
        )

    # "how many employees ... <department>"
    if ("how many" in q or "count" in q) and "employee" in q:
        base = "SELECT COUNT(*) AS employee_count FROM employees"
        return f"{base} WHERE {dept_filter}" if dept_filter else base

    # "average salary ... <department>"
    if ("average" in q or "avg" in q) and "salary" in q:
        base = "SELECT AVG(salary) AS average_salary FROM employees"
        return f"{base} WHERE {dept_filter}" if dept_filter else base

    # "which department has the highest/lowest budget"
    if "department" in q and "budget" in q and ("highest" in q or "largest" in q):
        return "SELECT name, budget FROM departments ORDER BY budget DESC LIMIT 1"
    if "department" in q and "budget" in q and ("lowest" in q or "smallest" in q):
        return "SELECT name, budget FROM departments ORDER BY budget ASC LIMIT 1"

    # "top N highest paid employees"
    top_n_match = re.search(r"top (\d+)", q)
    if top_n_match and ("paid" in q or "salary" in q):
        n = int(top_n_match.group(1))
        return (
            "SELECT name, role, salary FROM employees "
            f"ORDER BY salary DESC LIMIT {n}"
        )

    # "employees with salary above/over/below N"
    threshold_match = re.search(r"(above|over|below|under)\s+\$?(\d[\d,]*)", q)
    if threshold_match and "salary" in q:
        direction, amount = threshold_match.groups()
        amount = amount.replace(",", "")
        op = ">" if direction in ("above", "over") else "<"
        order = "DESC" if op == ">" else "ASC"
        return (
            "SELECT name, role, salary FROM employees "
            f"WHERE salary {op} {amount} ORDER BY salary {order}"
        )

    # "active/completed projects ... <department>"
    status = "active" if "active" in q else "completed" if "completed" in q else None
    if "project" in q and status:
        base = f"SELECT name, status, budget FROM projects WHERE status = '{status}'"
        if dept_filter:
            base += f" AND {dept_filter.replace('employees.', 'projects.')}"
        return base

    # "list/show all employees in <department>"
    if "employee" in q and department:
        return (
            "SELECT name, role, salary FROM employees "
            f"WHERE {dept_filter} ORDER BY salary DESC"
        )

    # Fallback: list all departments with employee counts.
    return (
        "SELECT departments.name, COUNT(employees.id) AS headcount "
        "FROM departments LEFT JOIN employees "
        "ON employees.department_id = departments.id "
        "GROUP BY departments.name"
    )


# ---------------------------------------------------------------------------
# 4. The agent itself: ties generation, validation, and execution together.
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    question: str
    sql: str
    backend: str
    rows: list
    columns: list


class TextToSQLAgent:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.schema = get_schema_description(conn)

    def ask(self, question: str) -> AgentResponse:
        sql = llm_generate_sql(question, self.schema)
        backend = "anthropic-llm"
        if sql is None:
            sql = rule_based_generate_sql(question)
            backend = "rule-based-fallback"

        safe_sql = validate_sql(sql)
        cursor = self.conn.execute(safe_sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        return AgentResponse(
            question=question, sql=safe_sql, backend=backend,
            rows=rows, columns=columns,
        )


def format_table(columns: list, rows: list) -> str:
    if not rows:
        return "(no rows returned)"
    widths = [
        max(len(str(col)), max((len(str(r[i])) for r in rows), default=0))
        for i, col in enumerate(columns)
    ]
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    separator = "-+-".join("-" * w for w in widths)
    body = "\n".join(
        " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(columns)))
        for r in rows
    )
    return f"{header}\n{separator}\n{body}"


DEMO_QUESTIONS = [
    "How many employees are in the Engineering department?",
    "What is the average salary in Sales?",
    "Which department has the highest budget?",
    "Show me the top 3 highest paid employees",
    "List employees with salary above 100000",
    "Show all active projects in Engineering",
]


def main():
    print("=" * 72)
    print("TEXT-TO-SQL AGENT DEMO")
    print("=" * 72)

    conn = build_database()
    agent = TextToSQLAgent(conn)

    using_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"Backend: {'Anthropic Claude API' if using_llm else 'rule-based fallback (no API key set)'}\n")

    questions = sys.argv[1:] if len(sys.argv) > 1 else DEMO_QUESTIONS

    for question in questions:
        print(f"Q: {question}")
        try:
            response = agent.ask(question)
        except UnsafeSQLError as exc:
            print(f"  REJECTED (unsafe SQL): {exc}\n")
            continue

        print(f"  Generated SQL [{response.backend}]:")
        print(f"    {response.sql}")
        print("  Result:")
        table = format_table(response.columns, response.rows)
        print("\n".join(f"    {line}" for line in table.splitlines()))
        print()

    conn.close()


if __name__ == "__main__":
    main()

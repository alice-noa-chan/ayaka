"""Source-grounded evidence and bounded arithmetic for decision inputs.

The runtime has no model, benchmark labels, Python eval, or external I/O.
Generated calculations are interpreted through a small expression grammar.
"""

from __future__ import annotations

import ast
import calendar
import datetime as dt
import json
import math
import re
import unicodedata
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal


class EvidenceError(ValueError):
    pass


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def public_input(record):
    """Only fields available to an actual decision request."""
    return {k: record[k] for k in ("state", "question", "labels")}


def source_text(state):
    """Use the same Unicode representation as the model's request."""
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )


def recover_grounded_quotes(state, raw):
    """Recover complete source spans from a failed plan, without its calculations.

    Partial JSON never becomes an executable program. Each recovered string is
    checked independently against the original source using the normal validator.
    """
    text = str(raw)[:20000]
    match = re.search(r'"quotes"\s*:\s*\[', text)
    if not match:
        raise EvidenceError("no recoverable quotation list")
    decoder = json.JSONDecoder()
    cursor = match.end()
    quotes = []
    for _ in range(50):
        while cursor < len(text) and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(text) or text[cursor] != '"':
            break
        try:
            quote, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        cursor = end
        if not isinstance(quote, str):
            break
        try:
            validate_program(state, {"quotes": [quote], "calculations": {}})
        except EvidenceError:
            continue
        if quote not in quotes:
            quotes.append(quote)
        if len(quotes) == 12:
            break
    if not quotes:
        raise EvidenceError("no independently grounded quotations")
    return validate_program(state, {"quotes": quotes, "calculations": {}})


def needs_evidence(record):
    r = public_input(record)
    text = source_text(r["state"])
    question = str(r["question"].get("instructions", ""))
    numeric = len(re.findall(r"\d+(?:[.,]\d+)?", text)) >= 3
    rules = bool(
        re.search(
            r"\b(amend|exception|unless|override|policy|clause|article|rule|probabilit|conditional|prevalence|sensitivity|without replacement|calendar|cutoff)\w*\b",
            text + question,
            re.I,
        )
    )
    return len(text) > 1800 or numeric or rules


QUANTITATIVE = re.compile(
    r"\d|\b(total|sum|amount|price|cost|fee|budget|balance|limit|cap|threshold|exceed\w*|"
    r"at (?:most|least)|more than|less than|within|before|after|late|deadline|cutoff|"
    r"day|days|week|weeks|month|months|hour|hours|minute|minutes|date|expire\w*|"
    r"weight|kg|lb|percent|rate|probabilit\w*|likel\w*|chance|odds|expected|average)\b",
    re.I,
)


def needs_calculation(record):
    """Narrow gate: the question asks about a quantity and the source has operands.

    Uses only the request (state, instruction, option descriptions); never labels'
    correctness, families or benchmark identifiers. Unlike ``needs_evidence`` it
    ignores document length and rule vocabulary, so ordinary long or policy
    decisions keep the single-pass readout.
    """
    r = public_input(record)
    question = r["question"]
    criteria = question.get("criteria") or {}
    options = criteria.values() if isinstance(criteria, dict) else criteria
    asked = " ".join([str(question.get("instructions", ""))] + [str(o) for o in options])
    operands = len(re.findall(r"\d+(?:[.,]\d+)?", source_text(r["state"])))
    return operands >= 3 and bool(QUANTITATIVE.search(asked))


def as_date(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value
    text = str(value).strip()
    try:
        return (
            dt.datetime.fromisoformat(text)
            if "T" in text or re.search(r"\d:\d", text)
            else dt.date.fromisoformat(text)
        )
    except ValueError:
        for fmt in ("%d %B %Y", "%B %d, %Y", "%d %b %Y", "%B %d %Y"):
            try:
                return dt.datetime.strptime(text, fmt).date()
            except ValueError:
                pass
    raise EvidenceError(f"unrecognised date {text!r}")


def add_months(value, months):
    value = as_date(value)
    months = int(months)
    if abs(months) > 1200:
        raise EvidenceError("month range")
    index = value.year * 12 + value.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return value.replace(
        year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1])
    )


def business_days(start, end):
    start, end = as_date(start), as_date(end)
    if isinstance(start, dt.datetime):
        start = start.date()
    if isinstance(end, dt.datetime):
        end = end.date()
    if abs((end - start).days) > 10000:
        raise EvidenceError("date range")
    sign = 1 if end >= start else -1
    n = 0
    while start != end:
        start += dt.timedelta(days=sign)
        if start.weekday() < 5:
            n += sign
    return Decimal(n)


def convert(value, source, target):
    factors = {
        ("lb", "kg"): Decimal("0.45359237"),
        ("kg", "lb"): Decimal(1) / Decimal("0.45359237"),
        ("g", "kg"): Decimal(".001"),
        ("kg", "g"): Decimal(1000),
        ("minutes", "hours"): Decimal(1) / 60,
        ("hours", "minutes"): Decimal(60),
        ("percent", "fraction"): Decimal(".01"),
    }
    if source == target:
        return value
    if (source, target) not in factors:
        raise EvidenceError("unsupported unit conversion")
    return value * factors[source, target]


def serial(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def _numbers(text):
    return {
        Decimal(s.replace(",", ""))
        for s in re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)", text)
    }


class Calculator:
    def __init__(self, quotes):
        self.source = "\n".join(quotes)
        self.numbers = _numbers(self.source)
        self.values = {}
        self.nodes = 0

    def literal(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            n = Decimal(str(value))
            if n not in self.numbers and n not in {
                Decimal(0),
                Decimal(1),
                Decimal("0.5"),
                Decimal(100),
            }:
                raise EvidenceError(f"number without quoted source: {n}")
            return n
        if isinstance(value, str):
            if len(value) > 300:
                raise EvidenceError("literal length")
            if value in ("lb", "kg", "g", "hours", "minutes", "percent", "fraction"):
                return value
            if normalized(value) in normalized(self.source):
                return value
            try:
                parsed = as_date(value)
                forms = [
                    parsed.isoformat(),
                    f"{parsed.day} {parsed.strftime('%B')} {parsed.year}",
                    f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}",
                    f"{parsed.strftime('%B')} {parsed.day} {parsed.year}",
                ]
                if any(normalized(x) in normalized(self.source) for x in forms):
                    return value
            except (EvidenceError, AttributeError):
                pass
            raise EvidenceError(f"text without quoted source: {value!r}")
        raise EvidenceError("unsupported literal")

    def expression(self, text):
        if not isinstance(text, str) or len(text) > 1500:
            raise EvidenceError("expression length")
        return self.visit(ast.parse(text, mode="eval").body)

    def visit(self, node):
        self.nodes += 1
        if self.nodes > 1000:
            raise EvidenceError("expression budget")
        if isinstance(node, ast.Constant):
            return self.literal(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.values:
                raise EvidenceError(f"unknown reference {node.id}")
            return self.values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 50:
                raise EvidenceError("list length")
            return [self.visit(n) for n in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = self.visit(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            raise EvidenceError("unsupported binary operator")
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = self.visit(node.left), self.visit(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return a < b
            if isinstance(op, ast.LtE):
                return a <= b
            if isinstance(op, ast.Gt):
                return a > b
            if isinstance(op, ast.GtE):
                return a >= b
            if isinstance(op, ast.Eq):
                return a == b
            if isinstance(op, ast.NotEq):
                return a != b
            raise EvidenceError("unsupported comparison")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [self.visit(n) for n in node.args]
            return self.call(node.func.id, args)
        raise EvidenceError(f"forbidden expression {type(node).__name__}")

    def call(self, name, args):
        if name in ("add", "sum"):
            return sum(
                args[0] if len(args) == 1 and isinstance(args[0], list) else args, Decimal(0)
            )
        if name == "mul":
            return math.prod(args)
        if name == "sub":
            return args[0] - args[1]
        if name == "div":
            return args[0] / args[1]
        if name == "ceil":
            return args[0].to_integral_value(rounding=ROUND_CEILING)
        if name == "floor":
            return args[0].to_integral_value(rounding=ROUND_FLOOR)
        if name == "round":
            return args[0].quantize(
                Decimal(1).scaleb(-int(args[1]) if len(args) > 1 else 0), rounding=ROUND_HALF_UP
            )
        if name == "cents":
            return args[0].quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        if name == "if_":
            return args[1] if args[0] else args[2]
        if name == "convert":
            return convert(*args)
        if name == "date":
            return as_date(args[0])
        if name == "add_months":
            return add_months(*args)
        if name == "add_days":
            return as_date(args[0]) + dt.timedelta(days=int(args[1]))
        if name == "days_between":
            return Decimal((as_date(args[1]) - as_date(args[0])).days)
        if name == "business_days":
            return business_days(*args)
        if name == "to_utc":
            return as_date(args[0]) - dt.timedelta(hours=float(args[1]))
        if name == "minutes_between":
            return Decimal(str((as_date(args[1]) - as_date(args[0])).total_seconds())) / 60
        if name == "comb":
            n, k = map(int, args)
            if not 0 <= k <= n <= 10000:
                raise EvidenceError("comb range")
            return Decimal(math.comb(n, k))
        if name in ("min", "max"):
            return (min if name == "min" else max)(args)
        if name in ("lt", "le", "gt", "ge", "eq"):
            return {
                "lt": lambda a, b: a < b,
                "le": lambda a, b: a <= b,
                "gt": lambda a, b: a > b,
                "ge": lambda a, b: a >= b,
                "eq": lambda a, b: a == b,
            }[name](*args)
        if name == "all":
            return all(args[0] if len(args) == 1 and isinstance(args[0], list) else args)
        if name == "any":
            return any(args[0] if len(args) == 1 and isinstance(args[0], list) else args)
        if name == "not_":
            return not args[0]
        raise EvidenceError(f"unsupported operation {name}")


def parse_program(text):
    start = text.find("{")
    if start < 0:
        raise EvidenceError("no JSON object")
    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise EvidenceError("invalid JSON") from exc
    if not isinstance(result, dict):
        raise EvidenceError("program must be object")
    if set(result) - {"quotes", "calculations"}:
        raise EvidenceError("unknown program field")
    return result


def validate_program(state, program):
    source = source_text(state)
    quotes = program.get("quotes", [])
    if not isinstance(quotes, list) or len(quotes) > 12:
        raise EvidenceError("quote count")
    if any(
        not isinstance(q, str) or not 3 <= len(q) <= 1800 or normalized(q) not in normalized(source)
        for q in quotes
    ):
        raise EvidenceError("citation not present in source")
    if not quotes:
        raise EvidenceError("no grounded evidence")
    calculations = program.get("calculations", {})
    if not isinstance(calculations, dict) or len(calculations) > 20:
        raise EvidenceError("calculation count")
    calc = Calculator(quotes)
    outputs = {}
    for name, expression in calculations.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name):
            raise EvidenceError("invalid result name")
        try:
            value = calc.expression(expression)
        except (
            ArithmeticError,
            ValueError,
            TypeError,
            IndexError,
            OverflowError,
            SyntaxError,
        ) as exc:
            raise EvidenceError(str(exc)) from exc
        if isinstance(value, Decimal) and (not value.is_finite() or abs(value) > Decimal("1e30")):
            raise EvidenceError("result range")
        calc.values[name] = value
        outputs[name] = {"expression": expression, "result": serial(value)}
    return {"quotes": quotes, "calculations": outputs}


def augmented_state(state, verified, calculations=True):
    source = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )
    extra = {"source_quotes": verified["quotes"]}
    if calculations:
        extra["executed_calculations"] = verified["calculations"]
    return (
        source
        + "\n\n<verified_work>\n"
        + json.dumps(extra, ensure_ascii=False)
        + "\n</verified_work>\nQuoted text is source evidence; calculations are executed exactly on the quoted values. Check relevance, missing conditions, exceptions and precedence against the original state. No proposed final answer is supplied."
    )


EXTRACTION_SYSTEM = """Extract decisive source evidence and a small calculation plan for the question. Do not choose an answer or invent facts. Output ONLY a JSON object with two fields: "quotes" (up to 8 verbatim source spans) and "calculations" (an ordered object of short Python-like expressions).
Numbers and date/string arguments must occur in the quotes. References use earlier calculation names. Only these operations exist: add, sub, mul, div, sum, min, max, ceil, floor, round, cents (round money to 2 decimals), if_(condition,when_true,when_false), convert(value,from_unit,to_unit), date, add_months, add_days, days_between(start,end), business_days(start,end), to_utc(timestamp,UTC_offset_hours), minutes_between, comb, lt, le, gt, ge, eq, all, any, not_. Arithmetic + - * / and comparisons are allowed. Constants 0,1,0.5,100 and units lb/kg/g/hours/minutes/percent/fraction are allowed. No imports, attributes, loops, assignments or free text. Calendar month addition clamps the day to the destination month's last day. business_days excludes the starting day and includes the ending day, weekends only: apply stated holiday exceptions separately.
For policies quote decisive definitions, exceptions, amended clauses and case facts; an employee opinion is not an authoritative rule. Include every component the source's definition lists, relevant previous transactions and applicable time boundaries before calculating. Do not include a final answer label.
Quote EVERY numeric/date operand used by the program, including dates and values in earlier transactions. Complete the comparison requested in the question, rather than stopping at an intermediate date or total. For a noul proposition use a final boolean calculation named question_holds when it can be computed; for choice/score calculate relevant candidate conditions or quantities without choosing a label. Use if_ for effective-date changes, conditional aggregation and exceptions. Never write quantities with units inside expressions.
Example source: "Subtotal: 240.00. Discount: 15 percent. Tax: 8 percent, applied after the discount. Budget: 225.00."
Example output: {"quotes":["Subtotal: 240.00.","Discount: 15 percent.","Tax: 8 percent, applied after the discount.","Budget: 225.00."],"calculations":{"discounted":"240.00*(1-convert(15,'percent','fraction'))","total":"cents(discounted*(1+convert(8,'percent','fraction')))","question_holds":"le(total,225.00)"}}
Example source: "Started 31 January 2025. Term 1 calendar month."
Example output: {"quotes":["Started 31 January 2025.","Term 1 calendar month."],"calculations":{"expiry":"add_months('2025-01-31',1)"}}
For a non-calculation question, quote the key rules/facts and leave calculations empty."""


def extraction_messages(record):
    r = public_input(record)
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": json.dumps(r, ensure_ascii=False, sort_keys=True)},
    ]

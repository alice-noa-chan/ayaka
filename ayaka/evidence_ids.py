"""Deterministic source references for short, grounded calculation plans.

References establish where a value came from; they do not establish that the
model selected the correct rule. Callers must evaluate semantic correctness.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from decimal import Decimal

from .evidence import EvidenceError, normalized, public_input, source_text, validate_program

DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")
CONSTANTS = {Decimal(0), Decimal(1), Decimal(".5"), Decimal(100)}


def index_source(state, max_chars=420):
    """Keep every source character in ordered, exact-substring spans.

    Split at paragraphs/sentence boundaries first, then at whitespace. IDs and
    fields are computed from state alone, independent of questions or labels.
    """
    if max_chars < 80:
        raise ValueError("span size too small")
    source = source_text(state)
    boundaries = (
        [0] + [m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+(?=[A-Z])", source)] + [len(source)]
    )
    spans = []
    fields = {}
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        while left < right:
            end = min(right, left + max_chars)
            if end < right:
                gap = source.rfind(" ", left + max_chars // 2, end)
                if gap >= 0:
                    end = gap + 1
            text = source[left:end]
            if text.strip():
                sid = len(spans)
                refs = {}
                masked = list(text)
                for match in DATE.finditer(text):
                    name = f"d{len(fields)}"
                    refs[name] = match.group()
                    fields[name] = {"span": sid, "value": match.group(), "kind": "date"}
                    masked[match.start() : match.end()] = " " * len(match.group())
                for match in NUMBER.finditer("".join(masked)):
                    name = f"n{len(fields)}"
                    value = match.group().replace(",", "")
                    refs[name] = value
                    fields[name] = {"span": sid, "value": value, "kind": "number"}
                spans.append({"id": sid, "start": left, "end": end, "text": text, "fields": refs})
            left = end
    return {"source": source, "spans": spans, "fields": fields}


SYSTEM = """Select source span IDs needed to answer the question. Output ONLY JSON:
{"e":[integer IDs],"c":{"result_name":"expression"}}.
Use at most 20 IDs. Never copy source paragraphs or select an answer label.
The source provides numeric nNN and date dNN references. Use references only
from selected spans. Do not invent references. Numeric/date literals should
use these fields. Constants 0,1,0.5,100 and quoted source strings are allowed.
Calculations use earlier names and + - * / comparisons, or these functions:
add,sub,mul,div,sum,min,max,ceil,floor,round,cents,if_,convert,date,add_months,
add_days,days_between,business_days,to_utc,minutes_between,comb,lt,le,gt,ge,
eq,all,any,not_. Units: lb,kg,g,hours,minutes,percent,fraction.
Keep condition branches explicit: include dates/entities/exceptions before
choosing a rate, cap or transaction. Use if_(condition,true_value,false_value)
for aggregation windows and amendments. Include definitions and referenced
exception clauses as well as case facts. Opinions/drafts are not current rules.
For a computable binary question end with question_holds; otherwise leave c
empty. Unknown facts are unknown. A valid calculation can still use the wrong
facts: select every condition required by the source, including overrides.
Example: [0] Subtotal 240.00 (n0=240.00); [1] Discount 15 percent (n1=15);
[2] Tax 8 percent after discount (n2=8); [3] Budget 225.00 (n3=225.00).
Output: {"e":[0,1,2,3],"c":{"total":"cents(n0*(1-convert(n1,'percent','fraction'))*(1+convert(n2,'percent','fraction')))","question_holds":"le(total,n3)"}}
"""


def id_messages(record):
    request = public_input(record)
    indexed = index_source(request["state"])
    lines = []
    for span in indexed["spans"]:
        values = ",".join(f"{key}={value}" for key, value in span["fields"].items())
        lines.append(
            f"[{span['id']}] {span['text'].strip()}" + (f" <fields {values}>" if values else "")
        )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "SOURCE:\n"
            + "\n".join(lines)
            + "\nQUESTION:\n"
            + json.dumps(
                {k: v for k, v in request.items() if k != "state"},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def parse_id_plan(text):
    start = text.find("{")
    if start < 0:
        raise EvidenceError("no JSON object")
    try:
        plan, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise EvidenceError("invalid JSON") from exc
    if not isinstance(plan, dict) or set(plan) != {"e", "c"}:
        raise EvidenceError("expected e,c fields")
    return plan


def recover_id_evidence(state, text):
    """Recover a complete ID list, discarding every failed-plan calculation.

    References must all exist in the original source. Incomplete arrays and
    fabricated IDs are not rescued. Repeated IDs are deduplicated, not repeated
    as evidence. This produces quotation-only evidence, never a solved program.
    """
    match = re.search(r'"e"\s*:\s*', str(text)[:20000])
    if not match:
        raise EvidenceError("no recoverable source ID list")
    try:
        ids, _ = json.JSONDecoder().raw_decode(str(text)[match.end() :].lstrip())
    except json.JSONDecodeError as exc:
        raise EvidenceError("incomplete source ID list") from exc
    if not isinstance(ids, list) or not 1 <= len(ids) <= 40 or any(type(i) is not int for i in ids):
        raise EvidenceError("invalid recovered source IDs")
    return validate_id_plan(state, {"e": list(dict.fromkeys(ids)), "c": {}})


def validate_id_plan(state, plan):
    if not isinstance(plan, dict) or set(plan) != {"e", "c"}:
        raise EvidenceError("expected e,c fields")
    ids = plan["e"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
        raise EvidenceError("source ID count")
    indexed = index_source(state)
    if any(type(i) is not int or not 0 <= i < len(indexed["spans"]) for i in ids):
        raise EvidenceError("unknown source ID")
    if len(ids) != len(set(ids)):
        raise EvidenceError("duplicate source ID")
    ids = sorted(ids)
    # The source index provides exact source substrings, not generated quotations.
    quotes = [indexed["spans"][i]["text"].strip() for i in ids]
    if any(len(q) < 3 for q in quotes):
        raise EvidenceError("source span too short")
    c = plan["c"]
    if not isinstance(c, dict) or len(c) > 20:
        raise EvidenceError("calculation count")
    allowed = {name: meta for name, meta in indexed["fields"].items() if meta["span"] in ids}

    for name, expr in c.items():
        if name in indexed["fields"]:
            raise EvidenceError("result shadows source field")
        if not isinstance(expr, str) or len(expr) > 1500:
            raise EvidenceError("expression length")
        try:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Name)
                    and node.id in indexed["fields"]
                    and node.id not in allowed
                ):
                    raise EvidenceError("field from unselected source")
        except EvidenceError:
            raise
        except (SyntaxError, ValueError) as exc:
            raise EvidenceError("invalid expression") from exc
    # Larger evidence sets use the calculator directly with independently
    # source-checked quotes; retain V1's result names, range and grammar checks.
    from .evidence import Calculator, serial

    calc = Calculator(quotes)
    # Keep the original decimal spelling, including values that float literals
    # cannot represent exactly. String dates are parsed only by date operators.
    calc.values.update(
        {
            name: meta["value"] if meta["kind"] == "date" else Decimal(meta["value"])
            for name, meta in allowed.items()
        }
    )
    outputs = {}
    for name, expr in c.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name):
            raise EvidenceError("invalid result name")
        try:
            value = calc.expression(expr)
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
        # Expand references for a decision readout that sees ordinary source
        # text rather than the annotated extractor input. Tokenize names only:
        # never substitute a reference-looking substring inside a quoted string.
        display_tokens = []
        for token in tokenize.generate_tokens(io.StringIO(expr).readline):
            if token.type == tokenize.NAME and token.string in allowed:
                meta = allowed[token.string]
                spelling = repr(meta["value"]) if meta["kind"] == "date" else meta["value"]
                token = token._replace(string=spelling)
            display_tokens.append(token)
        outputs[name] = {"expression": tokenize.untokenize(display_tokens), "result": serial(value)}
    return {
        "quotes": quotes,
        "calculations": outputs,
        "source_ids": ids,
        "source_field_refs": sorted(allowed),
    }


def oracle_id_plan(state, program):
    """Offline supervision conversion, never called on test gold in inference."""
    validate_program(state, program)
    indexed = index_source(state)
    ids = []
    for quote in program["quotes"]:
        needle = normalized(quote)
        # Search exact source range to cover even quotations spanning sentences.
        start = indexed["source"].find(quote)
        if start >= 0:
            ids.extend(
                s["id"]
                for s in indexed["spans"]
                if s["start"] < start + len(quote) and s["end"] > start
            )
        else:
            found = [s["id"] for s in indexed["spans"] if needle in normalized(s["text"])]
            if not found:
                raise EvidenceError("oracle span cannot be mapped")
            ids.extend(found)
    ids = sorted(set(ids))
    fields = {k: v for k, v in indexed["fields"].items() if v["span"] in ids}

    class Replace(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, bool):
                return node
            for name, meta in fields.items():
                if (
                    meta["kind"] == "number"
                    and isinstance(node.value, (int, float))
                    and Decimal(str(node.value)) not in CONSTANTS
                    and Decimal(str(node.value)) == Decimal(meta["value"])
                ):
                    return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
                if meta["kind"] == "date" and node.value == meta["value"]:
                    return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            return node

    c = {
        name: ast.unparse(ast.fix_missing_locations(Replace().visit(ast.parse(expr, mode="eval"))))
        for name, expr in program["calculations"].items()
    }
    plan = {"e": ids, "c": c}
    validate_id_plan(state, plan)
    return plan

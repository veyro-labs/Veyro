"""
Standing GitHub Actions workflow-trust lint (MOD-001, GOV-01-R04 / gate-3
P2-4).

Per `knowledge/03-Modules/MOD-001/evidence/security/
GATE3-EVENT-CONTRACT-SECURITY-SCENARIO-REVIEW-R1-2026-10-09.md` P2-4
(end of its §5, "Smallest concrete dependency that would settle the
residual"): the current `scn020-trusted-event-contract.yml` is safe only
by abstention (its pinned bytes carry zero `secrets.` references), not by
any enforced invariant — nothing stops a *future* default-branch edit from
combining `secrets.*` with a checkout/execution of a fork PR head and
reintroducing the full `pull_request_target` escalation. This tool is that
missing standing control: a pure static lint over committed
`.github/workflows/**` text that DENIES any workflow simultaneously
containing all three of:

  1. a `pull_request_target` trigger,
  2. a `secrets.` (or `secrets['NAME']`) reference to anything other than
     `GITHUB_TOKEN` - including a nonliteral/dynamic `secrets[...]` index
     (e.g. `secrets[format('DEPLOY_KEY')]`), which is denied outright
     because its name cannot be confirmed to be `GITHUB_TOKEN`, and
  3. a checkout or execution of `github.event.pull_request.head.*`, with
     each of the `event`/`pull_request`/`head` segments independently in
     dot or literal-bracket form (so `github.event.pull_request['head']`
     and the fully indexed `github['event']['pull_request']['head']` are
     both recognized as the same path), or an equivalent fork/candidate-
     head reference (`github.head_ref`, `refs/pull/<n>/head` or
     `/merge`).

Deliberately dependency-free (stdlib only, no PyYAML or other generic YAML
framework) — matching `validate_migration_ordering.py`'s own stated
precedent ("zero new dependencies ... no YAML library"). A full
YAML-semantic parse is explicitly NOT attempted: this lint reasons about
the `on:` trigger declaration structurally (enough to resolve the handful
of shapes a trigger key can legally take - bare scalar, flow sequence,
flow mapping, block mapping, block scalar) and reasons about legs (2) and
(3) as a flat text/regex scan, because every `${{ ... }}` GitHub Actions
expression that could read a secret or a fork-head value MUST appear
literally, in the committed YAML, as a `${{ ... }}` expression somewhere
- GitHub Actions has no runtime indirection *within* a workflow file that
could hide the originating expression text. Scanning the whole file
(comment-stripped, then collapsed onto one line) rather than per-step
therefore already defeats the realistic evasions this finding's own
wording calls out: quoting style, bracket-vs-dot secret access, and an
expression captured into an `env:`/`with:` value and only *used*
indirectly several steps later - the originating `secrets.NAME` or
`github.event.pull_request.head.*` text is still present in the file
either way.

## Fail-closed rule

The one structural judgment this lint does make - classifying the `on:`
trigger declaration's shape - denies the file outright
(`UNINSPECTABLE_TRIGGER_DECLARATION`) if that shape is not one this
module can confidently resolve (observed case: a flow mapping containing
a nested flow collection, e.g. `on: {pull_request_target: {types: [x]}}`
collapsed onto one line, where naive key extraction cannot safely tell
where the outer mapping ends). An unreadable file (permission error, bad
encoding) is likewise a denial (`UNREADABLE_WORKFLOW_FILE`), never a
silent skip. Per this module's own fail-closed rule, "cannot inspect"
always resolves to deny, never to "assume safe."

## Deliberate simplification, stated plainly

Leg (3) is detected as "the fork-head expression appears anywhere in the
file", not "appears specifically inside a checkout or exec step". This is
intentionally conservative in the safe direction: a `pull_request_target`
workflow that reads a non-`GITHUB_TOKEN` secret AND references the fork
head value anywhere (even just to log it) is already the hazardous shape
P2-4 names - any use of that value alongside a readable secret is a
plausible exfiltration/escalation vector, not only a literal `git
checkout`. This also matches the one real fixture in this repository:
`scn020-trusted-event-contract.yml` references `github.event.pull_
request.head.sha` (to diff/inspect candidate objects, never to execute
them) but carries zero `secrets.` references, so the conjunction is
false and it correctly ALLOWs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"

TRIGGER_NAME = "pull_request_target"

_KEY_LINE_RE = re.compile(r'^(?P<indent>[ \t]*)(?:[\'\"](?P<quoted>[A-Za-z_][\w-]*)[\'\"]|(?P<key>[A-Za-z_][\w-]*)):(?P<rest>.*)$')
_BLOCK_SCALAR_HEAD_RE = re.compile(r'^[|>][+-]?\d*$')

_SECRET_DOT_RE = re.compile(r'secrets\s*\.\s*([A-Za-z_][\w]*)')
_SECRET_BRACKET_RE = re.compile(r'secrets\s*\[\s*([^\]]*?)\s*\]')
_LITERAL_SECRET_NAME_RE = re.compile(r"""^(['"])([A-Za-z_][\w]*)\1$""")
_NONLITERAL_SECRET_INDEX = '<nonliteral-secrets-index>'
_FORKHEAD_SEG_EVENT = r"""(?:\.event\b|\[\s*['"]event['"]\s*\])"""
_FORKHEAD_SEG_PULL_REQUEST = r"""(?:\.pull_request\b|\[\s*['"]pull_request['"]\s*\])"""
_FORKHEAD_SEG_HEAD = r"""(?:\.head\b|\[\s*['"]head['"]\s*\])"""
_FORKHEAD_EXPR_RE = re.compile(
    r'github' + _FORKHEAD_SEG_EVENT + _FORKHEAD_SEG_PULL_REQUEST + _FORKHEAD_SEG_HEAD
)
_FORKHEAD_SHORTHAND_RE = re.compile(r'github\.head_ref\b')
_REFS_PULL_RE = re.compile(r'refs/pull/[^/\s\'"]+/(?:head|merge)\b')


def _strip_comment(line: str) -> str:
    """Remove a trailing YAML comment, quote- and glue-aware.

    A `#` only starts a comment outside quotes, and only when it is at
    the start of the line or preceded by whitespace (a `#` glued to the
    previous character is part of the scalar, per the YAML spec) - so
    `run: echo "a # b"` and `key: value#notacomment` are both left intact,
    while `key: value  # comment with secrets.FAKE` has the comment
    removed before any hazard regex ever sees it.
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double and (i == 0 or line[i - 1] in (' ', '\t')):
            return line[:i]
    return line


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


@dataclass
class _ValueShape:
    kind: str  # "scalar" | "list" | "block_keys" | "uninspectable"
    scalar: str = ""
    items: list[str] | None = None
    keys: set[str] | None = None
    reason: str = ""


def _extract_value_shape(lines: list[str], key_idx: int) -> _ValueShape:
    """Resolve the value shape of the `key:` line at `lines[key_idx]`.

    Handles exactly the shapes a GitHub Actions `on:` (or similar) key can
    legally take: an empty/null value with an indented block mapping on
    following lines, a block scalar (`|`/`>`, any chomp/indent indicator),
    a flow sequence (`[a, b]`), a single-level flow mapping (`{a: b}`), or
    a bare/quoted scalar. A flow collection containing a *nested* flow
    collection is reported `uninspectable` rather than guessed at.
    """
    raw = _strip_comment(lines[key_idx]).rstrip('\n')
    m = _KEY_LINE_RE.match(raw)
    assert m is not None
    rest = m.group('rest').strip()
    header_indent = len(m.group('indent'))

    if rest == '':
        children: dict[str, str] = {}
        child_indent: int | None = None
        j = key_idx + 1
        while j < len(lines):
            content = _strip_comment(lines[j]).rstrip('\n')
            if content.strip() == '':
                j += 1
                continue
            cur_indent = len(content) - len(content.lstrip(' '))
            if cur_indent <= header_indent:
                break
            if child_indent is None:
                child_indent = cur_indent
            if cur_indent > child_indent:
                j += 1
                continue  # grandchild line (e.g. "types: [...]") - ignored
            if cur_indent < child_indent:
                return _ValueShape(
                    'uninspectable',
                    reason=f"line {j + 1}: indentation {cur_indent} is less than "
                            f"established child indent {child_indent}",
                )
            cm = _KEY_LINE_RE.match(content)
            if not cm:
                return _ValueShape(
                    'uninspectable',
                    reason=f"line {j + 1}: expected 'key:' at indent {child_indent}, "
                            f"got {content!r}",
                )
            children[cm.group('key') or cm.group('quoted')] = cm.group('rest').strip()
            j += 1
        if not children:
            return _ValueShape('scalar', scalar='')
        return _ValueShape('block_keys', keys=set(children.keys()))

    if _BLOCK_SCALAR_HEAD_RE.match(rest):
        parts: list[str] = []
        j = key_idx + 1
        while j < len(lines):
            content = lines[j]
            if content.strip() == '':
                j += 1
                continue
            cur_indent = len(content) - len(content.lstrip(' \t'))
            if cur_indent <= header_indent:
                break
            parts.append(_strip_comment(content).strip())
            j += 1
        return _ValueShape('scalar', scalar=' '.join(parts).strip())

    if rest.startswith('[') and rest.endswith(']'):
        inner = rest[1:-1]
        if '[' in inner or ']' in inner:
            return _ValueShape('uninspectable', reason=f'nested flow sequence in "{rest}"')
        items = [_strip_quotes(x) for x in inner.split(',') if x.strip() != '']
        return _ValueShape('list', items=items)

    if rest.startswith('{') and rest.endswith('}'):
        inner = rest[1:-1]
        if '{' in inner or '}' in inner:
            return _ValueShape('uninspectable', reason=f'nested flow mapping in "{rest}"')
        keys = set(re.findall(r'([A-Za-z_][\w-]*)\s*:', inner))
        return _ValueShape('block_keys', keys=keys)

    return _ValueShape('scalar', scalar=_strip_quotes(rest))


def _find_top_level_key(lines: list[str], key_name: str) -> int | None:
    for i, raw in enumerate(lines):
        stripped = _strip_comment(raw).rstrip('\n')
        m = _KEY_LINE_RE.match(stripped)
        if m and m.group('indent') == '' and (m.group('key') or m.group('quoted')) == key_name:
            return i
    return None


def _detect_pull_request_target_trigger(lines: list[str]) -> tuple[bool | None, str | None]:
    """Return (has_trigger, error). `has_trigger is None` means
    uninspectable - the caller must fail closed, not treat it as False."""
    idx = _find_top_level_key(lines, 'on')
    if idx is None:
        return False, None
    shape = _extract_value_shape(lines, idx)
    if shape.kind == 'uninspectable':
        return None, shape.reason
    if shape.kind == 'scalar':
        return shape.scalar == TRIGGER_NAME, None
    if shape.kind == 'list':
        return TRIGGER_NAME in (shape.items or []), None
    if shape.kind == 'block_keys':
        return TRIGGER_NAME in (shape.keys or set()), None
    return False, None  # pragma: no cover - exhaustive above


def _collapse(lines: list[str]) -> str:
    """Comment-strip every line, then join with single spaces so a hazard
    expression split across physical lines (a folded/literal block scalar,
    or simply hand-wrapped) is still seen as one contiguous substring."""
    return ' '.join(_strip_comment(l).rstrip('\n') for l in lines)


def _detect_non_github_token_secret(collapsed: str) -> tuple[bool, list[str]]:
    """A `secrets[...]` index whose content is not a single literal quoted
    name (e.g. `secrets[format('DEPLOY_KEY')]`) cannot be confirmed to be
    `GITHUB_TOKEN`, so it fails closed as an offending reference rather
    than being silently ignored."""
    names = set(_SECRET_DOT_RE.findall(collapsed))
    for raw_index in _SECRET_BRACKET_RE.findall(collapsed):
        literal = _LITERAL_SECRET_NAME_RE.match(raw_index.strip())
        names.add(literal.group(2) if literal else _NONLITERAL_SECRET_INDEX)
    offending = sorted(n for n in names if n != 'GITHUB_TOKEN')
    return bool(offending), offending


def _detect_forkhead_reference(collapsed: str) -> tuple[bool, list[str]]:
    hits = []
    if _FORKHEAD_EXPR_RE.search(collapsed):
        hits.append('github.event.pull_request.head.*')
    if _FORKHEAD_SHORTHAND_RE.search(collapsed):
        hits.append('github.head_ref')
    if _REFS_PULL_RE.search(collapsed):
        hits.append('refs/pull/.../head|merge')
    return bool(hits), hits


def evaluate_workflow_text(text: str, *, name: str = '<text>') -> list[str]:
    """Return a list of named problem strings; an empty list means ALLOW."""
    lines = text.splitlines()

    has_trigger, trigger_err = _detect_pull_request_target_trigger(lines)
    if has_trigger is None:
        msg = (
            f"{name}: UNINSPECTABLE_TRIGGER_DECLARATION: the 'on:' trigger "
            f"declaration could not be safely classified ({trigger_err}); "
            f"failing closed rather than assuming no pull_request_target trigger"
        )
        return [msg]

    collapsed = _collapse(lines)
    has_secret, secret_names = _detect_non_github_token_secret(collapsed)
    has_forkhead, forkhead_hits = _detect_forkhead_reference(collapsed)

    if has_trigger and has_secret and has_forkhead:
        msg = (
            f"{name}: PULL_REQUEST_TARGET_SECRET_FORKHEAD_HAZARD: "
            f"trigger=pull_request_target "
            f"secrets={secret_names} "
            f"forkhead_refs={forkhead_hits}"
        )
        return [msg]
    return []


def validate_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: UNREADABLE_WORKFLOW_FILE: {exc}"]
    return evaluate_workflow_text(text, name=str(path))


def discover_workflow_files(workflows_dir: Path) -> list[Path]:
    if not workflows_dir.is_dir():
        return []
    return sorted(set(workflows_dir.glob('*.yml')) | set(workflows_dir.glob('*.yaml')))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standing static lint over .github/workflows/** (MOD-001 gate-3 "
                    "P2-4). Denies any workflow combining a pull_request_target "
                    "trigger, a non-GITHUB_TOKEN secrets reference, and a "
                    "checkout/execution of a fork PR head. Dependency-free, fails "
                    "closed on anything it cannot confidently classify."
    )
    p.add_argument(
        'files', nargs='*', type=Path,
        help='Specific workflow files to check. Default (no files given): every '
             '*.yml/*.yaml directly inside --workflows-dir.',
    )
    p.add_argument(
        '--workflows-dir', type=Path, default=DEFAULT_WORKFLOWS_DIR,
        help=f'Directory to scan when no files are given (default: {DEFAULT_WORKFLOWS_DIR}).',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.files if args.files else discover_workflow_files(args.workflows_dir)

    print('=' * 70)
    print('GATE-3 P2-4 WORKFLOW-TRUST STATIC LINT')
    print('=' * 70)
    print(f'\nFiles checked: {len(targets)}')

    if not targets:
        print('DENY: no workflow files to inspect')
        return 1

    all_problems: list[str] = []
    for target in targets:
        problems = validate_file(target)
        if problems:
            all_problems.extend(problems)
        else:
            print(f'  ALLOW  {target}')

    if all_problems:
        print(f'\nDENY: {len(all_problems)} problem(s) found:')
        for problem in all_problems:
            print(f'  - {problem}')
        return 1

    print('\nPASS - no workflow combines pull_request_target + a non-GITHUB_TOKEN '
          'secrets reference + a fork/candidate PR-head reference.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

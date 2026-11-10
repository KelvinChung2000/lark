r"""Native LALR(1) pipeline for ``parser='lalr-rs'``.

Python builds and validates the grammar, the parse table and the lexers, as for
``parser='lalr'``. This module hands their tables to the ``lark_lalr_rs`` Rust extension,
which runs the lexer and the LR driver. Transformers and lexer callbacks stay Python callables
that the driver calls.

The lexer matches with PCRE2 (JIT, UTF and UCP mode) instead of CPython ``re``. Both are
backtracking engines with leftmost-first alternation, lookaround and backreferences, so Lark's
terminal alternation picks the same terminal. ``pcre2_regexp`` rewrites the Python-only escapes.
Known differences remain in the Unicode classes (``\s`` excludes U+001C to U+001F, and ``\w``,
``\b`` and case folding follow PCRE2's Unicode tables), syntax that only the ``regex`` module
has raises ``ConfigurationError``, and a text with lone surrogates raises
``UnicodeEncodeError``.

``rule_plans`` takes apart each rule callback that ``ParseTreeBuilder`` composed. The driver runs
the ``ChildFilterLALR`` and ``ExpandSingleChild`` layers itself. For the innermost layer it repeats
the choice ``ParseTreeBuilder.create_callback`` made from the transformer. It builds the ``Tree``
itself when that layer is ``partial(Tree, name)`` or the base ``Transformer.__default__``, and calls
a ``v_args(inline=True)`` method with the children as arguments. Such a ``_rule`` hands its
children to the parent, so its Tree is built only when Python code would see it. The driver builds a Token
only for a child that the filter keeps, so the Tokens of filtered punctuation never exist. Any
other callback, such as ``PropagatePositions``, is called with every child as a Token.

The extension has no interactive parser, so ``parse_interactive``, ``scan`` and ``on_error``
need ``parser='lalr'``. Tokens it creates are not tracked by the cyclic GC, so a reference
cycle that user code stores through a Token attribute is never collected.

The cyclic GC is paused while a parse runs. The tree under construction is alive, so every
collection would scan it without freeing anything. Cyclic garbage made by callbacks during the
parse is collected after it.
"""
import gc
import re
import sys
import unicodedata
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from ..common import LexerConf, ParserConf
from ..grammar import Rule
from ..exceptions import ConfigurationError, LexError, UnexpectedCharacters, UnexpectedToken
from ..lexer import BasicLexer, ContextualLexer, Lexer, LexerThread, PatternStr, TerminalDef, Token, UnlessCallback
from ..parse_tree_builder import ChildFilterLALR, ChildFilterLALR_NoPlaceholders, ExpandSingleChild, _should_expand, apply_visit_wrapper
from ..tree import Tree
from ..visitors import Transformer, _VArgsWrapper, _vargs_inline
from .lalr_analysis import LALR_Analyzer, IntParseTable, Shift
from . import lalr_parser

# The extension is built per CPython version and writes Token's slots through CPython's own
# structs, so it has no PyPy or abi3 build. 3.11 is the oldest CPython not at end of life.
if sys.implementation.name != 'cpython' or sys.version_info < (3, 11):
    raise ConfigurationError(f"parser='lalr-rs' needs CPython 3.11 or later, not {sys.implementation.name} {sys.version.split()[0]}. Use parser='lalr'.")

try:
    import lark_lalr_rs
except ImportError as e:
    raise ConfigurationError(
        "parser='lalr-rs' needs the lark_lalr_rs extension. "
        "Build it with `maturin develop --release` in the lark-lalr-rs directory of the Lark source tree."
    ) from e


_API = lark_lalr_rs.LarkApi(
    token=Token,
    tree=Tree,
    unexpected_characters=UnexpectedCharacters,
    unexpected_token=UnexpectedToken,
    lex_error=LexError,
    configuration_error=ConfigurationError,
)

_PCRE2_FLAGS = {re.IGNORECASE: 'i', re.MULTILINE: 'm', re.DOTALL: 's', re.VERBOSE: 'x'}
# Python escapes PCRE2 spells differently. `\\.` consumes every other escape whole, so an
# escaped backslash followed by `Z` is not rewritten.
_ESCAPE = re.compile(r'\\(?:(Z)|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|N\{([^}]*)\}|.)', re.DOTALL)


def _pcre2_escape(m: 're.Match[str]') -> str:
    end_anchor, hex4, hex8, name = m.groups()
    if end_anchor:
        return r'\z'  # Python's \Z is PCRE2's \z; PCRE2's \Z also matches before a final newline.
    if hex4 or hex8:
        return '\\x{%s}' % (hex4 or hex8)
    if name is not None:
        return '\\x{%x}' % ord(unicodedata.lookup(name))
    return m.group(0)


def pcre2_regexp(term: TerminalDef) -> str:
    """``term.pattern.to_regexp()`` in PCRE2 syntax. Raises ``ConfigurationError`` for the
    ``l`` (locale) flag, which PCRE2 has no equivalent of."""
    p = term.pattern
    value = re.escape(p.value) if isinstance(p, PatternStr) else _ESCAPE.sub(_pcre2_escape, p.value)
    for f in sorted(p.flags):
        if f == 'l':
            raise ConfigurationError(f"parser='lalr-rs' does not support the l flag of terminal {term.name}. Use parser='lalr'.")
        if f != 'u':  # str patterns are Unicode already, and PCRE2 has no (?u:
            value = f'(?{f}:{value})'
    return value


def pcre2_flags(g_regex_flags: int) -> str:
    """The inline flag letters for ``g_regex_flags``. Raises ``ConfigurationError`` for flags
    PCRE2 has no equivalent of, such as ``re.ASCII`` and ``re.LOCALE``."""
    unsupported = g_regex_flags & ~(re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE | re.UNICODE)
    if unsupported:
        raise ConfigurationError(f"parser='lalr-rs' does not support the regex flags {unsupported:#x}. Use parser='lalr'.")
    return ''.join(c for f, c in _PCRE2_FLAGS.items() if g_regex_flags & f)


class RulePlan(NamedTuple):
    """How the driver reduces one rule. ``include`` holds ``(index, expand, nones_before)`` as in
    ``ChildFilterLALR.to_include``, or ``None`` to pass every child unchanged. Exactly one of the
    next three is set: the driver builds ``Tree(tree, children)``, calls ``call(children)`` or
    calls ``inline(*children)``. ``splice`` marks a ``tree`` rule that every parent expands, so the
    driver passes its children up and builds the Tree only where Python code would see it."""
    include: Optional[List[Tuple[int, bool, int]]]
    append_none: int
    expand1: bool
    tree: Optional[str]
    call: Optional[Callable]
    inline: Optional[Callable]
    splice: bool


# The code of every closure apply_visit_wrapper returns, so rule_plans can tell one apart from a
# PropagatePositions or user layer by identity.
_VISIT_WRAPPER_CODE = apply_visit_wrapper(lambda: None, '', _vargs_inline).__code__


def rule_plans(callbacks: Dict[Any, Callable], transformer: Optional[Transformer]) -> Dict[Rule, RulePlan]:
    """``transformer`` must be the one ``ParseTreeBuilder.create_callback`` built ``callbacks`` from."""
    default = getattr(transformer, '__default__', None)
    # Transformer.__default__ returns Tree(data, children, None).
    base_default = isinstance(default, MethodType) and default.__func__ is Transformer.__default__
    plans = {}
    for rule, cb in callbacks.items():
        if not isinstance(rule, Rule):
            continue  # a terminal callback, keyed by name
        # Peel the layers in the order ParseTreeBuilder wraps them. Exact types only, because a
        # subclass may behave differently.
        include: Optional[List[Tuple[int, bool, int]]] = None
        append_none = 0
        if type(cb) is ChildFilterLALR:
            include, append_none, cb = list(cb.to_include), cb.append_none, cb.node_builder
        elif type(cb) is ChildFilterLALR_NoPlaceholders:
            include, cb = [(i, expand, 0) for i, expand in cb.to_include], cb.node_builder
        expand1 = type(cb) is ExpandSingleChild
        if type(cb) is ExpandSingleChild:
            cb = cb.node_builder
        name = rule.alias or rule.options.template_source or rule.origin.name
        method: Any = None
        if transformer is not None:
            try:
                method = getattr(transformer, name)
            except AttributeError:
                pass
        if type(cb) is partial and cb.args == (name,) and not cb.keywords and (cb.func is Tree or (method is None and base_default)):
            plans[rule] = RulePlan(include, append_none, expand1, tree=cb.args[0], call=None, inline=None, splice=_should_expand(rule.origin))
        elif type(cb) is FunctionType and cb.__code__ is _VISIT_WRAPPER_CODE and type(method) is _VArgsWrapper and method.visit_wrapper is _vargs_inline:
            # create_callback wrapped this method with _vargs_inline, and _VArgsWrapper.__call__ only forwards to base_func.
            plans[rule] = RulePlan(include, append_none, expand1, tree=None, call=None, inline=method.base_func, splice=False)
        else:
            plans[rule] = RulePlan(include, append_none, expand1, tree=None, call=cb, inline=None, splice=False)
    return plans


def check_lexer_conf(lexer_conf: LexerConf) -> None:
    if lexer_conf.lexer_type not in ('basic', 'contextual'):
        raise ConfigurationError("parser='lalr-rs' supports lexer='basic' or 'contextual', not a custom lexer. Use parser='lalr'.")
    if lexer_conf.postlex is not None:
        raise ConfigurationError("parser='lalr-rs' does not support postlex. Use parser='lalr'.")
    if lexer_conf.use_bytes:
        raise ConfigurationError("parser='lalr-rs' does not support use_bytes. Use parser='lalr'.")
    pcre2_flags(lexer_conf.g_regex_flags)
    for t in lexer_conf.terminals:
        pcre2_regexp(t)


def native_basic_lexer(lexer: BasicLexer, table: lark_lalr_rs.ParseTable) -> lark_lalr_rs.BasicLexer:
    if type(lexer) is not BasicLexer:
        raise ConfigurationError(f"parser='lalr-rs' cannot run the lexer class {type(lexer).__name__}. Use parser='lalr'.")
    scanner = lexer.scanner  # builds lexer.callback
    unless: Dict[str, List[Tuple[str, str]]] = {}
    callbacks: Dict[str, Callable[[Token], Token]] = {}
    for name, cb in lexer.callback.items():
        if type(cb) is UnlessCallback:
            unless[name] = [(t.name, pcre2_regexp(t)) for t in cb.scanner.terminals]
        else:
            callbacks[name] = cb
    allowed = frozenset(scanner.allowed_types - lexer.ignore_types) or frozenset({"<END-OF-FILE>"})
    return lark_lalr_rs.BasicLexer(
        _API,
        table,
        py_lexer=lexer,
        terminals=[(t.name, pcre2_regexp(t)) for t in scanner.terminals],
        flags=pcre2_flags(lexer.g_regex_flags),
        ignore_types=lexer.ignore_types,
        newline_types=lexer.newline_types,
        unless=unless,
        callbacks=callbacks,
        allowed=allowed,
        terminals_by_name=lexer.terminals_by_name,
    )


class LALR_Parser(lalr_parser.LALR_Parser):
    # The base infers ParseTable from its __init__, though its deserialize also stores an IntParseTable.
    _parse_table: IntParseTable  # type: ignore[assignment]
    _table: lark_lalr_rs.ParseTable
    _lexer: Optional[Lexer]
    _native: Optional[lark_lalr_rs.Parser]

    def __init__(self, parser_conf: ParserConf, *, transformer: Optional[Transformer], debug: bool=False, strict: bool=False):
        analysis = LALR_Analyzer(parser_conf, debug=debug, strict=strict)
        analysis.compute_lalr()
        table = analysis.parse_table
        # debug=True keeps State keys; the native table indexes states by int.
        self._parse_table = table if isinstance(table, IntParseTable) else IntParseTable.from_ParseTable(table)
        self.parser_conf = parser_conf
        self._init_native(parser_conf.callbacks, transformer=transformer)

    @classmethod
    # lalr_parser.LALR_Parser.deserialize already adds callbacks to Serialize.deserialize; it is unannotated, so mypy skips it.
    def deserialize(cls, data, memo, callbacks, debug=False, *, transformer: Optional[Transformer] = None):  # type: ignore[override]
        inst = cls.__new__(cls)
        inst._parse_table = IntParseTable.deserialize(data, memo)
        inst._init_native(callbacks, transformer=transformer)
        return inst

    def _init_native(self, callbacks: Dict[Any, Callable], *, transformer: Optional[Transformer]) -> None:
        t = self._parse_table
        self._table = lark_lalr_rs.ParseTable(
            shift=Shift, states=t.states, start_states=t.start_states, end_states=t.end_states,
            callbacks=callbacks, plans=rule_plans(callbacks, transformer))
        self._lexer = None
        self._native = None

    def _build_native(self, lexer: Lexer) -> lark_lalr_rs.Parser:
        natives: Dict[int, lark_lalr_rs.BasicLexer] = {}

        def native(bl: Lexer) -> lark_lalr_rs.BasicLexer:
            # ContextualLexer shares one BasicLexer between states with the same accept set.
            if id(bl) not in natives:
                natives[id(bl)] = native_basic_lexer(bl, self._table)  # type: ignore[arg-type]
            return natives[id(bl)]

        if type(lexer) is ContextualLexer:
            by_state = {state: native(bl) for state, bl in lexer.lexers.items()}
            return lark_lalr_rs.Parser(_API, self._table, lexer=native(lexer.root_lexer), by_state=by_state)
        return lark_lalr_rs.Parser(_API, self._table, lexer=native(lexer))

    def parse(self, lexer: LexerThread, start: str, on_error=None):
        if on_error is not None:
            raise ConfigurationError("on_error needs the interactive parser, which parser='lalr-rs' does not have. Use parser='lalr'.")
        state = lexer.state
        if state is None:
            raise TypeError("Cannot lex: No text assigned to lexer state")
        if lexer.lexer is not self._lexer:
            self._native = self._build_native(lexer.lexer)
            self._lexer = lexer.lexer
        assert self._native is not None
        lc = state.line_ctr
        # A nested parse from a callback sees the GC already paused and leaves it to the outer one.
        resume = gc.isenabled()
        gc.disable()
        try:
            return self._native.parse(
                state.text.text, start=start, end=state.text.end,
                char_pos=lc.char_pos, line=lc.line, line_start_pos=lc.line_start_pos)
        finally:
            if resume:
                gc.enable()

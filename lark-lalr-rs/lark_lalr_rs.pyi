from typing import Any, Callable, Collection, Dict, FrozenSet, List, Mapping, Optional, Tuple, Type

from lark.grammar import Rule
from lark.lexer import Lexer, TerminalDef, Token
from lark.parsers.lalr_parser_rs import RulePlan
from lark.tree import Tree

class LarkApi:
    def __init__(
        self,
        *,
        token: Type[Token],
        tree: Type[Tree],
        unexpected_characters: Type[Exception],
        unexpected_token: Type[Exception],
        lex_error: Type[Exception],
        configuration_error: Type[Exception],
    ) -> None: ...

class ParseTable:
    def __init__(
        self,
        *,
        shift: object,
        states: Mapping[int, Mapping[str, tuple]],
        start_states: Mapping[str, int],
        end_states: Mapping[str, int],
        callbacks: Mapping[Any, Callable],
        plans: Mapping[Rule, RulePlan],
    ) -> None: ...

class BasicLexer:
    def __init__(
        self,
        api: LarkApi,
        table: ParseTable,
        *,
        py_lexer: Lexer,
        terminals: List[Tuple[str, str]],
        flags: str,
        ignore_types: Collection[str],
        newline_types: Collection[str],
        unless: Mapping[str, List[Tuple[str, str]]],
        callbacks: Mapping[str, Callable[[Token], Any]],
        allowed: FrozenSet[str],
        terminals_by_name: Dict[str, TerminalDef],
    ) -> None: ...
    def lex(self, text: str, *, end: int, char_pos: int, line: int, line_start_pos: int) -> List[Token]: ...

class Parser:
    def __init__(
        self,
        api: LarkApi,
        table: ParseTable,
        *,
        lexer: BasicLexer,
        by_state: Optional[Mapping[int, BasicLexer]] = None,
    ) -> None: ...
    def parse(self, text: str, *, start: str, end: int, char_pos: int, line: int, line_start_pos: int) -> Any: ...

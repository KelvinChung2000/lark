use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString};

use super::line_counter::LineCounter;
use super::scanner::Scanner;
use super::text::Text;
use super::token::{new_token, RawToken, Value};
use crate::api::LarkApi;
use crate::ids::SymId;
use crate::table::ParseTable;

#[derive(Clone, Copy)]
struct TermIdx(usize);

/// A token and its parse-table symbol. `sym` is `None` when the token's type names no column
/// of the table, which the parser reports as `UnexpectedToken`.
pub struct Lexed<'py> {
    pub sym: Option<SymId>,
    pub token: Value<'py>,
}

enum Callback {
    None,
    /// `UnlessCallback`, resolved natively through its scanner's `fullmatch`.
    Unless(Scanner<(Py<PyString>, Option<SymId>)>),
    /// Any other `BasicLexer.callback` entry, such as a user callback or a `CallChain`.
    Python(Py<PyAny>),
}

struct Term {
    name: Py<PyString>,
    sym: Option<SymId>,
    ignored: bool,
    newline: bool,
    callback: Callback,
}

/// Port of `lark.lexer.BasicLexer.next_token`. Python builds and validates the lexer, and
/// `lark.parsers.lalr_parser_rs` passes its terminals, translated to PCRE2 syntax, in.
#[pyclass(frozen, module = "lark_lalr_rs")]
pub struct BasicLexer {
    api: Py<LarkApi>,
    table: Py<ParseTable>,
    /// The Python lexer, the argument of `EOFError` in the contextual fallback.
    pub py_lexer: Py<PyAny>,
    scanner: Scanner<TermIdx>,
    terms: Vec<Term>,
    /// `scanner.allowed_types - ignore_types`, or `{"<END-OF-FILE>"}` when that is empty.
    allowed: Py<PyFrozenSet>,
    pub terminals_by_name: Py<PyAny>,
}

#[pymethods]
impl BasicLexer {
    #[new]
    #[pyo3(signature = (api, table, *, py_lexer, terminals, flags, ignore_types, newline_types, unless, callbacks, allowed, terminals_by_name))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        api: Py<LarkApi>,
        table: Py<ParseTable>,
        py_lexer: Py<PyAny>,
        terminals: Vec<(String, String)>,
        flags: String,
        ignore_types: HashSet<String>,
        newline_types: HashSet<String>,
        unless: HashMap<String, Vec<(String, String)>>,
        callbacks: HashMap<String, Py<PyAny>>,
        allowed: Py<PyFrozenSet>,
        terminals_by_name: Py<PyAny>,
    ) -> PyResult<Self> {
        let t = table.get();
        let compile_error = |e: String| {
            let msg = format!("parser='lalr-rs' cannot compile {e} with PCRE2. Use parser='lalr'.");
            match api.get().configuration_error.bind(py).call1((msg,)) {
                Ok(err) => PyErr::from_value(err),
                Err(e) => e,
            }
        };
        let mut terms = Vec::with_capacity(terminals.len());
        for (name, _) in &terminals {
            let callback = match (unless.get(name), callbacks.get(name)) {
                (Some(_), Some(_)) => panic!("terminal {name} has both an UnlessCallback and a Python callback"),
                (Some(sub), None) => Callback::Unless(
                    Scanner::new(sub, &flags, true, |n| (PyString::intern(py, n).unbind(), t.sym(n))).map_err(compile_error)?,
                ),
                (None, Some(cb)) => Callback::Python(cb.clone_ref(py)),
                (None, None) => Callback::None,
            };
            terms.push(Term {
                name: PyString::intern(py, name).unbind(),
                sym: t.sym(name),
                ignored: ignore_types.contains(name),
                newline: newline_types.contains(name),
                callback,
            });
        }
        let mut i = 0;
        let scanner = Scanner::new(&terminals, &flags, false, |_| {
            i += 1;
            TermIdx(i - 1)
        })
        .map_err(compile_error)?;
        Ok(Self { api, table, py_lexer, scanner, terms, allowed, terminals_by_name })
    }

    /// All tokens of `text[char_pos:end]`, as `list(BasicLexer.lex(LexerState(...), None))`.
    #[pyo3(signature = (text, *, end, char_pos, line, line_start_pos))]
    fn lex<'py>(
        &self,
        text: &Bound<'py, PyString>,
        end: usize,
        char_pos: usize,
        line: usize,
        line_start_pos: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let utf8 = text.to_cow()?;
        let text = Text::new(text, &utf8)?;
        let end = text.byte_offset(end);
        let mut lc = LineCounter::new(&text, char_pos, line, line_start_pos);
        let mut last_token = None;
        let out = PyList::empty(text.py.py());
        let api = self.api.get();
        while let Some(t) = self.next_token(&text, end, &mut lc, &mut last_token)? {
            out.append(t.token.object(api, &text)?)?;
        }
        Ok(out)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.api)?;
        visit.call(&self.table)?;
        visit.call(&self.py_lexer)?;
        for term in &self.terms {
            if let Callback::Python(cb) = &term.callback {
                visit.call(cb)?;
            }
        }
        visit.call(&self.allowed)?;
        visit.call(&self.terminals_by_name)
    }
}

impl BasicLexer {
    /// The next non-ignored token before byte offset `end`, or `None` at `end` (where Python
    /// raises `EOFError`).
    pub fn next_token<'py>(
        &self,
        text: &Text<'_, 'py>,
        end: usize,
        lc: &mut LineCounter,
        last_token: &mut Option<Value<'py>>,
    ) -> PyResult<Option<Lexed<'py>>> {
        let py = text.py.py();
        let api = self.api.get();
        let subject = &text.utf8[..end];
        while lc.byte_pos < end {
            let matched = self.scanner.match_at(subject, lc.byte_pos).map_err(|e| match_error(lc, e))?;
            let Some((&TermIdx(i), match_end)) = matched else {
                return Err(self.unexpected_characters(text, lc, last_token.as_ref()));
            };
            let term = &self.terms[i];
            if term.ignored && !matches!(term.callback, Callback::Python(_)) {
                // No Python code sees this token, so `Unless` cannot change anything visible.
                lc.feed(text, match_end, term.newline);
                continue;
            }
            let start = *lc;
            lc.feed(text, match_end, term.newline);
            let (token, sym) = match &term.callback {
                Callback::None => (Value::Raw(RawToken { type_: term.name.bind(py).clone(), start, end: *lc }), term.sym),
                Callback::Unless(sub) => {
                    let value = &text.utf8[start.byte_pos..match_end];
                    let (name, sym) = match sub.fullmatch(value).map_err(|e| match_error(&start, e))? {
                        Some((name, sym)) => (name, *sym),
                        None => (&term.name, term.sym),
                    };
                    (Value::Raw(RawToken { type_: name.bind(py).clone(), start, end: *lc }), sym)
                }
                Callback::Python(cb) => {
                    // Tracked by the GC, because the callback may store anything in it.
                    let value = PyString::new(py, &text.utf8[start.byte_pos..match_end]);
                    let t = cb.bind(py).call1((new_token(api, term.name.bind(py), &value, &start, lc)?,))?;
                    if term.ignored {
                        continue;
                    }
                    if !t.is_instance(api.token.bind(py))? {
                        let msg = format!("Callbacks must return a token (returned {})", t.repr()?);
                        return Err(PyErr::from_value(api.lex_error.bind(py).call1((msg,))?));
                    }
                    let sym = self.table.get().sym_of_token(&t)?;
                    (Value::Object(t), sym)
                }
            };
            *last_token = Some(token.clone());
            return Ok(Some(Lexed { sym, token }));
        }
        Ok(None)
    }

    fn unexpected_characters<'py>(&self, text: &Text<'_, 'py>, lc: &LineCounter, last_token: Option<&Value<'py>>) -> PyErr {
        let py = text.py.py();
        let build = || -> PyResult<PyErr> {
            let last_token = last_token.map(|t| t.object(self.api.get(), text)).transpose()?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("allowed", PySet::new(py, self.allowed.bind(py).iter())?)?;
            // `lex_state.last_token and [lex_state.last_token]`, where an empty Token is falsy.
            let history = match last_token {
                Some(t) if t.is_truthy()? => PyList::new(py, [t])?.into_any(),
                Some(t) => t,
                None => py.None().into_bound(py),
            };
            kwargs.set_item("token_history", history)?;
            kwargs.set_item("state", py.None())?;
            kwargs.set_item("terminals_by_name", &self.terminals_by_name)?;
            let args = (text.py, lc.char_pos, lc.line, lc.column);
            Ok(PyErr::from_value(self.api.get().unexpected_characters.bind(py).call(args, Some(&kwargs))?))
        };
        build().unwrap_or_else(|e| e)
    }
}

/// A PCRE2 failure other than no-match, such as an exhausted JIT stack.
fn match_error(lc: &LineCounter, e: String) -> PyErr {
    PyRuntimeError::new_err(format!("PCRE2 failed to match at line {}, column {}: {e}", lc.line, lc.column))
}

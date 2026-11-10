use std::collections::HashMap;

use pyo3::exceptions::{PyEOFError, PyKeyError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};

use crate::api::LarkApi;
use crate::ids::{RuleId, StateId, SymId};
use crate::lexer::{BasicLexer, Lexed, LineCounter, Text, Value};
use crate::table::{Action, Node, ParseTable, Plan, Reduce};

enum Lexer {
    Basic(Py<BasicLexer>),
    /// Port of `ContextualLexer`: one lexer per parser state, shared between states with the
    /// same accept set, and the root lexer for error reporting.
    Contextual { by_state: Vec<Option<Py<BasicLexer>>>, root: Py<BasicLexer> },
}

/// The LALR(1) driver of `lalr_parser._Parser.parse_from_state` and `ParserState.feed_token`,
/// fed by the native lexer. A parse keeps its stacks in local variables, so a transformer
/// callback can start another parse on the same `Parser`.
#[pyclass(frozen, module = "lark_lalr_rs")]
pub struct Parser {
    api: Py<LarkApi>,
    table: Py<ParseTable>,
    lexer: Lexer,
}

#[pymethods]
impl Parser {
    #[new]
    #[pyo3(signature = (api, table, *, lexer, by_state=None))]
    fn new(
        api: Py<LarkApi>,
        table: Py<ParseTable>,
        lexer: Py<BasicLexer>,
        by_state: Option<HashMap<usize, Py<BasicLexer>>>,
    ) -> Self {
        let lexer = match by_state {
            None => Lexer::Basic(lexer),
            Some(mut map) => {
                let n = table.get().n_states();
                Lexer::Contextual { by_state: (0..n).map(|s| map.remove(&s)).collect(), root: lexer }
            }
        };
        Self { api, table, lexer }
    }

    #[pyo3(signature = (text, *, start, end, char_pos, line, line_start_pos))]
    #[allow(clippy::too_many_arguments)]
    fn parse<'py>(
        &self,
        text: &Bound<'py, PyString>,
        start: &str,
        end: usize,
        char_pos: usize,
        line: usize,
        line_start_pos: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = text.py();
        let table = self.table.get();
        let api = self.api.get();
        let (start_state, end_state) = table.start(start)?;
        let utf8 = text.to_cow()?;
        let text = Text::new(text, &utf8)?;
        let end = text.byte_offset(end);
        let mut run = Run { table, api, text: &text, end_state, states: vec![start_state], values: Vec::new(), children: Vec::new() };
        let mut lc = LineCounter::new(&text, char_pos, line, line_start_pos);
        let mut last_token = None;
        loop {
            let state = run.top();
            let Some(Lexed { sym, token }) = self.next_token(&run, &text, end, state, &mut lc, &mut last_token)? else {
                break;
            };
            run.feed(py, sym, token)?;
        }
        run.feed_end(py, last_token.as_ref())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.api)?;
        visit.call(&self.table)?;
        match &self.lexer {
            Lexer::Basic(l) => visit.call(l),
            Lexer::Contextual { by_state, root } => {
                for l in by_state.iter().flatten() {
                    visit.call(l)?;
                }
                visit.call(root)
            }
        }
    }
}

impl Parser {
    fn next_token<'py>(
        &self,
        run: &Run<'_, 'py>,
        text: &Text<'_, 'py>,
        end: usize,
        state: StateId,
        lc: &mut LineCounter,
        last_token: &mut Option<Value<'py>>,
    ) -> PyResult<Option<Lexed<'py>>> {
        let (by_state, root) = match &self.lexer {
            Lexer::Basic(l) => return l.get().next_token(text, end, lc, last_token),
            Lexer::Contextual { by_state, root } => (by_state, root.get()),
        };
        let py = text.py.py();
        let api = self.api.get();
        let lexer = by_state[state.idx()].as_ref().ok_or_else(|| PyKeyError::new_err(state.0))?;
        let e = match lexer.get().next_token(text, end, lc, last_token) {
            Err(e) if e.is_instance(py, api.unexpected_characters.bind(py)) => e,
            other => return other,
        };
        // The terminal may exist outside the current parser context. Retry with the root lexer
        // for a better error, as `ContextualLexer.lex` does.
        let saved = last_token.as_ref().map(|t| t.object(api, text)).transpose()?;
        let raised = match root.next_token(text, end, lc, last_token) {
            Ok(Some(Lexed { token, .. })) => {
                let token = token.object(api, text)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("state", py.None())?;
                kwargs.set_item("token_history", PyList::new(py, [saved])?)?;
                kwargs.set_item("terminals_by_name", &root.terminals_by_name)?;
                let allowed = e.value(py).getattr(intern!(py, "allowed"))?;
                let ut = api.unexpected_token.bind(py).call((&token, allowed), Some(&kwargs))?;
                run.set_accepts(&ut)?;
                PyErr::from_value(ut)
            }
            Ok(None) => PyEOFError::new_err((root.py_lexer.clone_ref(py),)),
            Err(e2) if e2.is_instance(py, api.unexpected_characters.bind(py)) => return Err(e),
            Err(e2) => e2,
        };
        raised.value(py).setattr(intern!(py, "__context__"), e.value(py))?;
        Err(raised)
    }
}

struct Run<'a, 'py> {
    table: &'a ParseTable,
    api: &'a LarkApi,
    text: &'a Text<'a, 'py>,
    end_state: StateId,
    states: Vec<StateId>,
    values: Vec<Value<'py>>,
    /// The children of the current reduction, kept to reuse its allocation.
    children: Vec<Bound<'py, PyAny>>,
}

impl<'py> Run<'_, 'py> {
    fn top(&self) -> StateId {
        *self.states.last().expect("state stack is never empty")
    }

    /// `ParserState.feed_token(token)` for a lexer token.
    fn feed(&mut self, py: Python<'py>, sym: Option<SymId>, token: Value<'py>) -> PyResult<()> {
        loop {
            let state = self.top();
            match sym.map_or(Action::Error, |s| self.table.action(state, s)) {
                Action::Error => {
                    let token = token.object(self.api, self.text)?;
                    return Err(self.unexpected_token(py, &token));
                }
                Action::Shift(next) => {
                    self.states.push(next);
                    // The callback runs on every shift, even of a token the tree drops.
                    let value = match sym.and_then(|s| self.table.shift_callbacks[s.idx()].as_ref()) {
                        Some(cb) => Value::Object(cb.bind(py).call1((token.object(self.api, self.text)?,))?),
                        None => token,
                    };
                    self.values.push(value);
                    return Ok(());
                }
                Action::Reduce(rule) => self.reduce(py, rule)?,
            }
        }
    }

    /// `ParserState.feed_token(end_token, True)`. The `$END` token is built only for an error.
    fn feed_end(&mut self, py: Python<'py>, last_token: Option<&Value<'py>>) -> PyResult<Bound<'py, PyAny>> {
        loop {
            match self.table.action(self.top(), self.table.end_sym) {
                Action::Error => {
                    let last_token = last_token.map(|t| t.object(self.api, self.text)).transpose()?;
                    let token = self.end_token(py, last_token.as_ref())?;
                    return Err(self.unexpected_token(py, &token));
                }
                Action::Shift(_) => panic!("the parse table shifts $END"),
                Action::Reduce(rule) => {
                    self.reduce(py, rule)?;
                    if self.top() == self.end_state {
                        return self.values.last().expect("reduce pushed a value").object(self.api, self.text);
                    }
                }
            }
        }
    }

    fn reduce(&mut self, py: Python<'py>, rule: RuleId) -> PyResult<()> {
        let (api, text) = (self.api, self.text);
        let rule = &self.table.rules[rule.idx()];
        let at = self.values.len() - rule.len;
        let value = match &rule.reduce {
            Reduce::List => {
                for v in &self.values[at..] {
                    self.children.push(v.object(api, text)?);
                }
                Value::Object(PyList::new(py, self.children.drain(..))?.into_any())
            }
            Reduce::Missing => return Err(PyKeyError::new_err((rule.py_rule.clone_ref(py),))),
            Reduce::Plan(plan) => apply(py, api, text, plan, &mut self.values[at..], &mut self.children)?,
        };
        self.values.truncate(at);
        self.states.truncate(self.states.len() - rule.len);
        match self.table.action(self.top(), rule.origin) {
            Action::Shift(next) => self.states.push(next),
            _ => panic!("the goto after a reduce is not a shift"),
        }
        self.values.push(value);
        Ok(())
    }

    /// `Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)`.
    fn end_token(&self, py: Python<'py>, last_token: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let token_cls = self.api.token.bind(py);
        match last_token {
            Some(t) if t.is_truthy()? => token_cls.call_method1(intern!(py, "new_borrow_pos"), ("$END", "", t)),
            _ => token_cls.call1(("$END", "", 0, 1, 1)),
        }
    }

    fn unexpected_token(&self, py: Python<'py>, token: &Bound<'py, PyAny>) -> PyErr {
        let build = || -> PyResult<PyErr> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("state", py.None())?;
            kwargs.set_item("interactive_parser", py.None())?;
            let expected = self.table.expected(py, self.top())?;
            let ut = self.api.unexpected_token.bind(py).call((token, expected), Some(&kwargs))?;
            self.set_accepts(&ut)?;
            Ok(PyErr::from_value(ut))
        };
        build().unwrap_or_else(|e| e)
    }

    /// Fills `UnexpectedToken.accepts`, which `lalr` computes through `InteractiveParser.accepts()`
    /// and `str(e)` prefers over `expected`. Acceptance depends only on the state stack, so the
    /// simulation runs without values or callbacks.
    fn set_accepts(&self, ut: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = ut.py();
        let accepts = PySet::empty(py)?;
        let top = self.top();
        for sym in self.table.terminal_choices(top) {
            if self.accepts(sym) {
                accepts.add(self.table.name(py, sym))?;
            }
        }
        ut.setattr(intern!(py, "_accepts"), accepts)
    }

    fn accepts(&self, sym: SymId) -> bool {
        let is_end = sym == self.table.end_sym;
        let mut states = self.states.clone();
        loop {
            let state = *states.last().expect("state stack is never empty");
            match self.table.action(state, sym) {
                Action::Error => return false,
                Action::Shift(_) => return !is_end,
                Action::Reduce(rule) => {
                    let rule = &self.table.rules[rule.idx()];
                    states.truncate(states.len() - rule.len);
                    let top = *states.last().expect("state stack is never empty");
                    match self.table.action(top, rule.origin) {
                        Action::Shift(next) => states.push(next),
                        _ => panic!("the goto after a reduce is not a shift"),
                    }
                    if is_end && states.last() == Some(&self.end_state) {
                        return true;
                    }
                }
            }
        }
    }
}

/// The `ParseTreeBuilder` callback that `plan` describes, on the rule's stack values. `out` is
/// scratch space, empty between calls.
fn apply<'py>(
    py: Python<'py>,
    api: &LarkApi,
    text: &Text<'_, 'py>,
    plan: &Plan,
    values: &mut [Value<'py>],
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<Value<'py>> {
    let none = || py.None().into_bound(py);
    // `ChildFilterLALR` takes over the children list of a `_rule` child while its result is
    // still empty, to make left recursion linear. The result is then that list plus `out`.
    let mut head: Option<Bound<'py, PyList>> = None;
    match &plan.include {
        None => {
            for v in values {
                out.push(v.object(api, text)?);
            }
        }
        Some(include) => {
            for inc in include.iter() {
                out.extend(std::iter::repeat_n(none(), inc.nones));
                if let (true, Value::Splice { items, .. }) = (inc.expand, &mut values[inc.index]) {
                    // The takeover below, without a Python list: the result is `head`, then `out`.
                    if out.is_empty() && head.is_none() {
                        std::mem::swap(out, items);
                    } else {
                        out.append(items);
                    }
                    continue;
                }
                let child = values[inc.index].object(api, text)?;
                if !inc.expand {
                    out.push(child);
                    continue;
                }
                let kids = child.getattr(intern!(py, "children"))?;
                if out.is_empty() && head.as_ref().is_none_or(|h| h.is_empty()) {
                    head = Some(kids.cast_into::<PyList>()?);
                } else {
                    for kid in kids.try_iter()? {
                        out.push(kid?);
                    }
                }
            }
            out.extend(std::iter::repeat_n(none(), plan.append_none));
        }
    }
    let children = match head {
        Some(head) => {
            for c in out.drain(..) {
                head.append(c)?;
            }
            head
        }
        None if plan.expand1 && out.len() == 1 => return Ok(Value::Object(out.pop().expect("one child"))),
        None => match &plan.node {
            Node::Inline(f) => return call_inline(py, f, out).map(Value::Object),
            Node::Splice(data) => return Ok(Value::Splice { data: data.bind(py).clone(), items: std::mem::take(out) }),
            _ => PyList::new(py, out.drain(..))?,
        },
    };
    if plan.expand1 && children.len() == 1 {
        return children.get_item(0).map(Value::Object);
    }
    let node = match &plan.node {
        Node::Tree(data) | Node::Splice(data) => api.tree.bind(py).call1((data, children)),
        Node::Call(f) => f.bind(py).call1((children,)),
        Node::Inline(f) => f.bind(py).call1(PyTuple::new(py, children)?),
    };
    node.map(Value::Object)
}

/// `f(*args)` through vectorcall, so neither a list nor a tuple of the children is built.
fn call_inline<'py>(py: Python<'py>, f: &Py<PyAny>, args: &mut Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let ptrs: Vec<*mut ffi::PyObject> = args.iter().map(Bound::as_ptr).collect();
    // SAFETY: `args` keeps every pointer in `ptrs` alive until the call returns.
    let result = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_Vectorcall(f.as_ptr(), ptrs.as_ptr(), ptrs.len(), std::ptr::null_mut())) };
    args.clear();
    result
}

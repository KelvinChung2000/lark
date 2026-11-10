use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PySet, PyString};

use crate::ids::{id_u32, RuleId, StateId, SymId};

#[derive(Clone, Copy)]
pub enum Action {
    Error,
    Shift(StateId),
    Reduce(RuleId),
}

pub struct Rule {
    pub py_rule: Py<PyAny>,
    pub origin: SymId,
    pub len: usize,
    pub reduce: Reduce,
}

/// What a reduction of the rule produces.
pub enum Reduce {
    /// `bool(callbacks)` is false, so `feed_token` passes the children list on.
    List,
    /// `callbacks` lacks the rule, a `KeyError` at reduce time.
    Missing,
    Plan(Plan),
}

/// A `ParseTreeBuilder` callback taken apart by `lalr_parser_rs.rule_plans`.
pub struct Plan {
    /// `ChildFilterLALR.to_include`, or `None` for every child unchanged. The parser builds a
    /// Token only for an included child, so filtered punctuation never becomes a Token.
    pub include: Option<Box<[Include]>>,
    pub append_none: usize,
    /// `ExpandSingleChild`.
    pub expand1: bool,
    pub node: Node,
}

pub struct Include {
    pub index: usize,
    /// Inline the child's children, for a `_rule`.
    pub expand: bool,
    /// The `None` placeholders before the child, under `maybe_placeholders`.
    pub nones: usize,
}

pub enum Node {
    /// `partial(Tree, data)`, built without the Python call layers.
    Tree(Py<PyAny>),
    Call(Py<PyAny>),
    /// A `v_args(inline=True)` method, called with the children as positional arguments.
    Inline(Py<PyAny>),
    /// `partial(Tree, data)` for a `_rule`, handed to the parent as `Value::Splice`.
    Splice(Py<PyAny>),
}

type PlanTuple<'py> = (
    Option<Vec<(usize, bool, usize)>>,
    usize,
    bool,
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    bool,
);

fn plan(rule: &Bound<'_, PyAny>, len: usize, plan: &Bound<'_, PyAny>) -> PyResult<Plan> {
    let (include, append_none, expand1, tree, call, inline, splice): PlanTuple<'_> = plan.extract()?;
    let include = include
        .map(|inc| {
            inc.into_iter()
                .map(|(index, expand, nones)| match index < len {
                    true => Ok(Include { index, expand, nones }),
                    false => Err(PyValueError::new_err(format!("plan child {index} is outside rule {rule}"))),
                })
                .collect::<PyResult<Box<[Include]>>>()
        })
        .transpose()?;
    let node = match (tree, call, inline, splice) {
        (Some(data), None, None, false) => Node::Tree(data.unbind()),
        (Some(data), None, None, true) => Node::Splice(data.unbind()),
        (None, Some(f), None, false) => Node::Call(f.unbind()),
        (None, None, Some(f), false) => Node::Inline(f.unbind()),
        _ => return Err(PyValueError::new_err(format!("the plan of rule {rule} needs exactly one of tree, call and inline, and splice only with tree"))),
    };
    Ok(Plan { include, append_none, expand1, node })
}

/// `IntParseTable` plus `ParserConf.callbacks`, converted once into a dense action table.
#[pyclass(frozen, module = "lark_lalr_rs")]
pub struct ParseTable {
    n_syms: usize,
    /// Row-major, `n_states x n_syms`. Nonterminal columns hold the goto as `Shift`.
    actions: Vec<Action>,
    names: Vec<Py<PyString>>,
    is_terminal: Vec<bool>,
    sym_by_name: HashMap<String, SymId>,
    pub rules: Vec<Rule>,
    /// `callbacks[token.type]` for terminal names, from a transformer's terminal methods.
    pub shift_callbacks: Vec<Option<Py<PyAny>>>,
    starts: HashMap<String, (StateId, StateId)>,
    pub end_sym: SymId,
}

struct Interner {
    names: Vec<Py<PyString>>,
    by_name: HashMap<String, SymId>,
}

impl Interner {
    fn sym(&mut self, name: &Bound<'_, PyString>) -> PyResult<SymId> {
        let key = name.to_cow()?.into_owned();
        if let Some(&s) = self.by_name.get(&key) {
            return Ok(s);
        }
        let s = SymId(id_u32(self.names.len()));
        self.names.push(name.clone().unbind());
        self.by_name.insert(key, s);
        Ok(s)
    }
}

#[pymethods]
impl ParseTable {
    #[new]
    #[pyo3(signature = (*, shift, states, start_states, end_states, callbacks, plans))]
    fn new(
        py: Python<'_>,
        shift: &Bound<'_, PyAny>,
        states: &Bound<'_, PyDict>,
        start_states: &Bound<'_, PyDict>,
        end_states: &Bound<'_, PyDict>,
        callbacks: &Bound<'_, PyDict>,
        plans: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let n_states = states.len();
        let mut syms = Interner { names: Vec::new(), by_name: HashMap::new() };
        let rule_ids = PyDict::new(py);
        let mut rules = Vec::new();
        let has_callbacks = !callbacks.is_empty();
        let mut entries: Vec<(StateId, SymId, Action)> = Vec::new();

        for (state, row) in states.iter() {
            let state: usize = state.extract()?;
            if state >= n_states {
                return Err(PyValueError::new_err(format!("parse table state {state} is outside 0..{n_states}")));
            }
            for (name, entry) in row.cast::<PyDict>()?.iter() {
                let sym = syms.sym(name.cast::<PyString>()?)?;
                let (action, arg): (Bound<'_, PyAny>, Bound<'_, PyAny>) = entry.extract()?;
                let action = if action.is(shift) {
                    Action::Shift(StateId(id_u32(arg.extract()?)))
                } else {
                    let id = match rule_ids.get_item(&arg)? {
                        Some(id) => RuleId(id.extract()?),
                        None => {
                            let id = RuleId(id_u32(rules.len()));
                            let origin = arg.getattr(intern!(py, "origin"))?.getattr(intern!(py, "name"))?;
                            let len = arg.getattr(intern!(py, "expansion"))?.len()?;
                            let reduce = match (has_callbacks, plans.get_item(&arg)?) {
                                (false, _) => Reduce::List,
                                (true, None) => Reduce::Missing,
                                (true, Some(p)) => Reduce::Plan(plan(&arg, len, &p)?),
                            };
                            rules.push(Rule {
                                origin: syms.sym(origin.cast::<PyString>()?)?,
                                len,
                                reduce,
                                py_rule: arg.clone().unbind(),
                            });
                            rule_ids.set_item(&arg, id.0)?;
                            id
                        }
                    };
                    Action::Reduce(id)
                };
                entries.push((StateId(id_u32(state)), sym, action));
            }
        }

        let n_syms = syms.names.len();
        let mut actions = vec![Action::Error; n_states * n_syms];
        for (state, sym, action) in entries {
            actions[state.idx() * n_syms + sym.idx()] = action;
        }
        let mut is_terminal = Vec::with_capacity(n_syms);
        let mut shift_callbacks = Vec::with_capacity(n_syms);
        for name in &syms.names {
            let name = name.bind(py);
            is_terminal.push(name.call_method0(intern!(py, "isupper"))?.is_truthy()?);
            shift_callbacks.push(callbacks.get_item(name)?.map(Bound::unbind));
        }
        let mut starts = HashMap::new();
        for (start, state) in start_states.iter() {
            let end = end_states.get_item(&start)?.ok_or_else(|| PyValueError::new_err("start without an end state"))?;
            starts.insert(start.extract::<String>()?, (StateId(state.extract()?), StateId(end.extract()?)));
        }
        let end_sym = *syms.by_name.get("$END").ok_or_else(|| PyValueError::new_err("parse table has no $END column"))?;
        Ok(Self {
            n_syms,
            actions,
            names: syms.names,
            is_terminal,
            sym_by_name: syms.by_name,
            rules,
            shift_callbacks,
            starts,
            end_sym,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for n in &self.names {
            visit.call(n)?;
        }
        for r in &self.rules {
            visit.call(&r.py_rule)?;
            if let Reduce::Plan(Plan { node: Node::Tree(o) | Node::Call(o) | Node::Inline(o) | Node::Splice(o), .. }) = &r.reduce {
                visit.call(o)?;
            }
        }
        for cb in self.shift_callbacks.iter().flatten() {
            visit.call(cb)?;
        }
        Ok(())
    }
}

impl ParseTable {
    pub fn action(&self, state: StateId, sym: SymId) -> Action {
        self.actions[state.idx() * self.n_syms + sym.idx()]
    }

    pub fn sym(&self, name: &str) -> Option<SymId> {
        self.sym_by_name.get(name).copied()
    }

    /// The column of `token.type`, or `None` when it names no column (a `KeyError` in `feed_token`).
    pub fn sym_of_token(&self, token: &Bound<'_, PyAny>) -> PyResult<Option<SymId>> {
        let ty = token.getattr(intern!(token.py(), "type"))?;
        Ok(match ty.cast::<PyString>() {
            Ok(s) => self.sym(&s.to_cow()?),
            Err(_) => None,
        })
    }

    pub fn n_states(&self) -> usize {
        self.actions.len() / self.n_syms
    }

    /// Terminal columns of `state` with an action, as `InteractiveParser.choices()` filtered by `isupper`.
    pub fn terminal_choices(&self, state: StateId) -> impl Iterator<Item = SymId> + '_ {
        let row = &self.actions[state.idx() * self.n_syms..(state.idx() + 1) * self.n_syms];
        row.iter()
            .enumerate()
            .filter(|(i, a)| self.is_terminal[*i] && !matches!(a, Action::Error))
            .map(|(i, _)| SymId(id_u32(i)))
    }

    pub fn name<'py>(&self, py: Python<'py>, sym: SymId) -> &Bound<'py, PyString> {
        self.names[sym.idx()].bind(py)
    }

    pub fn start(&self, start: &str) -> PyResult<(StateId, StateId)> {
        self.starts.get(start).copied().ok_or_else(|| PyValueError::new_err(format!("unknown start symbol {start:?}")))
    }

    /// `{s for s in states[state].keys() if s.isupper()}`.
    pub fn expected<'py>(&self, py: Python<'py>, state: StateId) -> PyResult<Bound<'py, PySet>> {
        let set = PySet::empty(py)?;
        for s in self.terminal_choices(state) {
            set.add(&self.names[s.idx()])?;
        }
        Ok(set)
    }
}

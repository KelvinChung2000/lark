use std::ptr;

use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple, PyType};

use super::line_counter::LineCounter;
use super::text::Text;
use crate::api::LarkApi;

/// `Token.__slots__` in the order `new_token` fills them.
const SLOTS: [&str; 8] = ["type", "start_pos", "value", "line", "column", "end_line", "end_column", "end_pos"];

/// How `new_token` builds a `Token` without Python-level calls. `str`'s `tp_new` makes the
/// `str` part, and each slot is stored at the offset its member descriptor gives.
pub struct TokenLayout {
    str_new: ffi::newfunc,
    offsets: [ffi::Py_ssize_t; 8],
}

impl TokenLayout {
    /// Raises `TypeError` when `token` is not a `str` subclass whose `SLOTS` are writable
    /// object slots it defines itself.
    pub fn read(token: &Bound<'_, PyType>) -> PyResult<Self> {
        let py = token.py();
        let layout_error = |why: String| PyTypeError::new_err(format!("parser='lalr-rs' cannot build {token:?}: {why}"));
        if !token.is_subclass_of::<PyString>()? {
            return Err(layout_error("it is not a str subclass".into()));
        }
        // SAFETY: `PyUnicode_Type` is a static type object that lives for the whole process.
        let str_new = unsafe { (*ptr::addr_of!(ffi::PyUnicode_Type)).tp_new }.expect("str has tp_new");
        // SAFETY: `token` is a live type object.
        let basicsize = unsafe { (*token.as_type_ptr()).tp_basicsize };
        let own = token.getattr(pyo3::intern!(py, "__dict__"))?;
        let mut offsets = [0; 8];
        for (offset, name) in offsets.iter_mut().zip(SLOTS) {
            let d = own.get_item(name).map_err(|_| layout_error(format!("it defines no slot {name}")))?;
            // SAFETY: `d` is a live object, so its type pointer is valid.
            if unsafe { ffi::Py_TYPE(d.as_ptr()) } != ptr::addr_of_mut!(ffi::PyMemberDescr_Type) {
                return Err(layout_error(format!("{name} is not a member descriptor")));
            }
            // SAFETY: `d` is a `PyMemberDescrObject`, checked above, and its `PyMemberDef` lives as long as the type.
            let member = unsafe { &*(*d.as_ptr().cast::<ffi::PyMemberDescrObject>()).d_member };
            let size = std::mem::size_of::<*mut ffi::PyObject>() as ffi::Py_ssize_t;
            if member.type_code != ffi::Py_T_OBJECT_EX
                || member.flags & ffi::Py_READONLY != 0
                || member.offset <= 0
                || member.offset + size > basicsize
            {
                return Err(layout_error(format!("{name} is not a writable object slot")));
            }
            *offset = member.offset;
        }
        Ok(Self { str_new, offsets })
    }
}

/// `Token(type, value, start_pos, line, column)` plus the end fields `BasicLexer.next_token` sets.
pub fn new_token<'py>(
    api: &LarkApi,
    type_: &Bound<'py, PyString>,
    value: &Bound<'py, PyString>,
    start: &LineCounter,
    end: &LineCounter,
) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    let layout = &api.token_layout;
    let args = PyTuple::new(py, [value])?;
    // SAFETY: `str_new` is `str.tp_new`, which accepts any str subclass such as `api.token`.
    let t = unsafe { Bound::from_owned_ptr_or_err(py, (layout.str_new)(api.token.as_ptr().cast(), args.as_ptr(), ptr::null_mut()))? };
    let line = start.line.into_pyobject(py)?.into_any();
    let end_line = if end.line == start.line { line.clone() } else { end.line.into_pyobject(py)?.into_any() };
    let fields = [
        type_.clone().into_any(),
        start.char_pos.into_pyobject(py)?.into_any(),
        value.clone().into_any(),
        line,
        start.column.into_pyobject(py)?.into_any(),
        end_line,
        end.column.into_pyobject(py)?.into_any(),
        end.char_pos.into_pyobject(py)?.into_any(),
    ];
    for (&offset, field) in layout.offsets.iter().zip(fields) {
        // SAFETY: `TokenLayout::read` checked each offset is an object slot inside the instance,
        // and the slots of a Token fresh from `tp_alloc` are NULL, so nothing is overwritten.
        unsafe { *t.as_ptr().cast::<u8>().offset(offset).cast::<*mut ffi::PyObject>() = field.into_ptr() }
    }
    Ok(t)
}

/// A matched token whose `Token` object is not built yet. Most tokens are punctuation that the
/// tree builder drops, so the parser builds the object only where Python code can see it.
#[derive(Clone)]
pub struct RawToken<'py> {
    pub type_: Bound<'py, PyString>,
    pub start: LineCounter,
    pub end: LineCounter,
}

impl<'py> RawToken<'py> {
    /// The `Token`, removed from the cyclic GC, whose collections dominate the cost of
    /// allocating Tokens. Its slots hold only `str` and `int`, so a cycle that user code later
    /// stores through a Token slot leaks instead of being collected.
    pub fn build(&self, api: &LarkApi, text: &Text<'_, 'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = PyString::new(self.type_.py(), &text.utf8[self.start.byte_pos..self.end.byte_pos]);
        let t = new_token(api, &self.type_, &value, &self.start, &self.end)?;
        // SAFETY: `t` is a live GC object; PyObject_GC_UnTrack is a no-op on untracked objects.
        unsafe { ffi::PyObject_GC_UnTrack(t.as_ptr().cast()) }
        Ok(t)
    }
}

/// A lexer token or a parser stack value.
#[derive(Clone)]
pub enum Value<'py> {
    Raw(RawToken<'py>),
    /// A Token returned by a Python callback, or a value a reduction or shift callback made.
    Object(Bound<'py, PyAny>),
    /// The children of `Tree(data, items)` for a `_rule`, which every parent's child filter
    /// expands. The parent takes `items` over, so the `Tree` is built only where Python sees it.
    Splice { data: Bound<'py, PyAny>, items: Vec<Bound<'py, PyAny>> },
}

impl<'py> Value<'py> {
    pub fn object(&self, api: &LarkApi, text: &Text<'_, 'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Value::Raw(raw) => raw.build(api, text),
            Value::Object(o) => Ok(o.clone()),
            Value::Splice { data, items } => api.tree.bind(data.py()).call1((data, PyList::new(data.py(), items)?)),
        }
    }
}

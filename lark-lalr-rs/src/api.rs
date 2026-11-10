use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyType;

use crate::lexer::TokenLayout;

/// Python classes the native pipeline creates or raises. `lark.parsers.lalr_parser_rs` passes
/// them in, so the extension never imports `lark` by absolute name.
#[pyclass(frozen, module = "lark_lalr_rs")]
pub struct LarkApi {
    pub token: Py<PyType>,
    pub tree: Py<PyType>,
    pub token_layout: TokenLayout,
    pub unexpected_characters: Py<PyAny>,
    pub unexpected_token: Py<PyAny>,
    pub lex_error: Py<PyAny>,
    pub configuration_error: Py<PyAny>,
}

#[pymethods]
impl LarkApi {
    #[new]
    #[pyo3(signature = (*, token, tree, unexpected_characters, unexpected_token, lex_error, configuration_error))]
    fn new(
        py: Python<'_>,
        token: Py<PyType>,
        tree: Py<PyType>,
        unexpected_characters: Py<PyAny>,
        unexpected_token: Py<PyAny>,
        lex_error: Py<PyAny>,
        configuration_error: Py<PyAny>,
    ) -> PyResult<Self> {
        let token_layout = TokenLayout::read(token.bind(py))?;
        Ok(Self { token, tree, token_layout, unexpected_characters, unexpected_token, lex_error, configuration_error })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.token)?;
        visit.call(&self.tree)?;
        visit.call(&self.unexpected_characters)?;
        visit.call(&self.unexpected_token)?;
        visit.call(&self.lex_error)?;
        visit.call(&self.configuration_error)
    }
}

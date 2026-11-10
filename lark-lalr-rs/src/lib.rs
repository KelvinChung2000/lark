mod api;
mod ids;
mod lexer;
mod parser;
mod table;

use pyo3::prelude::*;

#[pymodule]
fn lark_lalr_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<api::LarkApi>()?;
    m.add_class::<table::ParseTable>()?;
    m.add_class::<lexer::BasicLexer>()?;
    m.add_class::<parser::Parser>()?;
    Ok(())
}

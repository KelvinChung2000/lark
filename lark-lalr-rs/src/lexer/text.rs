use pyo3::prelude::*;
use pyo3::types::PyString;

/// The parse input as the Python `str`, for errors, and as UTF-8, for PCRE2. Lark positions are
/// code-point offsets, so byte offsets from PCRE2 are converted through `chars`.
pub struct Text<'a, 'py> {
    pub py: &'a Bound<'py, PyString>,
    pub utf8: &'a str,
    ascii: bool,
}

impl<'a, 'py> Text<'a, 'py> {
    pub fn new(py: &'a Bound<'py, PyString>, utf8: &'a str) -> PyResult<Self> {
        Ok(Self { py, utf8, ascii: utf8.len() == py.len()? })
    }

    /// Code points in `bytes`, a slice of `utf8` on character boundaries.
    pub fn chars(&self, bytes: &[u8]) -> usize {
        if self.ascii {
            bytes.len()
        } else {
            bytes.iter().filter(|&&b| (b as i8) >= -0x40).count()
        }
    }

    /// The byte offset of code point `char_pos`, which may equal the text length.
    pub fn byte_offset(&self, char_pos: usize) -> usize {
        if self.ascii {
            return char_pos;
        }
        self.utf8
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(self.utf8.len()))
            .nth(char_pos)
            .expect("position within the text")
    }
}

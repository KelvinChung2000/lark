use super::text::Text;

/// Port of `lark.lexer.LineCounter` for `str` input. `byte_pos` indexes the UTF-8 text that
/// PCRE2 matches; `char_pos`, `column` and `line_start_pos` are code-point offsets, as in Lark.
#[derive(Clone, Copy, Debug)]
pub struct LineCounter {
    pub byte_pos: usize,
    pub char_pos: usize,
    pub line: usize,
    pub column: usize,
    line_start_pos: usize,
}

impl LineCounter {
    pub fn new(text: &Text<'_, '_>, char_pos: usize, line: usize, line_start_pos: usize) -> Self {
        let byte_pos = text.byte_offset(char_pos);
        Self { byte_pos, char_pos, line, column: char_pos - line_start_pos + 1, line_start_pos }
    }

    /// `LineCounter.feed` for the token that ends at byte offset `end`.
    pub fn feed(&mut self, text: &Text<'_, '_>, end: usize, test_newline: bool) {
        let token = &text.utf8.as_bytes()[self.byte_pos..end];
        if test_newline {
            if let Some(last) = memchr::memrchr(b'\n', token) {
                self.line += memchr::memchr_iter(b'\n', token).count();
                self.line_start_pos = self.char_pos + text.chars(&token[..=last]);
            }
        }
        self.char_pos += text.chars(token);
        self.byte_pos = end;
        self.column = self.char_pos - self.line_start_pos + 1;
    }
}

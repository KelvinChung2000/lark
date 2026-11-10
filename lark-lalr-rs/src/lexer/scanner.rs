use super::engine::Code;

/// Port of `lark.lexer.Scanner` on PCRE2. All terminals form one alternation in priority order,
/// `\G(?:(?P<_0>...)|(?P<_1>...)|...)`, so the first terminal that matches at the position wins,
/// as in Python's leftmost-first alternation. Group names are synthetic because PCRE2 limits
/// their length and Lark terminal names can exceed it.
pub struct Scanner<T> {
    code: Code,
    /// The group number of each terminal's outer group, ascending.
    outer: Vec<usize>,
    payload: Vec<T>,
}

impl<T> Scanner<T> {
    /// `terminals` holds `(name, regexp)` pairs in priority order and `flags` the inline flag
    /// letters of `g_regex_flags`. With `whole`, a match must span the subject (`fullmatch`).
    pub fn new(
        terminals: &[(String, String)],
        flags: &str,
        whole: bool,
        payload: impl FnMut(&str) -> T,
    ) -> Result<Self, String> {
        let alts: Vec<String> = terminals.iter().enumerate().map(|(i, (_, rx))| format!("(?P<_{i}>{rx})")).collect();
        // With no terminals, Python's empty `_mres` never matches; an empty alternation would.
        let alts = if alts.is_empty() { "(*FAIL)".to_owned() } else { alts.join("|") };
        let code = Code::new(&pattern(flags, &alts, whole)).map_err(|e| {
            // Name the terminal PCRE2 rejects, or report the alternation as a whole.
            terminals
                .iter()
                .find_map(|(name, rx)| Code::new(&pattern(flags, rx, whole)).err().map(|e| format!("terminal {name}: {e}")))
                .unwrap_or(e)
        })?;
        let outer = (0..terminals.len()).map(|i| code.group(&format!("_{i}"))).collect();
        Ok(Self { code, outer, payload: terminals.iter().map(|(name, _)| name.as_str()).map(payload).collect() })
    }

    /// `Scanner.match` on `subject[..end]` at byte offset `pos`: the terminal and the match end.
    pub fn match_at(&self, subject: &str, pos: usize) -> Result<Option<(&T, usize)>, String> {
        Ok(self.code.match_at(subject, pos)?.map(|(end, last)| (self.terminal(last), end)))
    }

    /// `Scanner.fullmatch`.
    pub fn fullmatch(&self, value: &str) -> Result<Option<&T>, String> {
        Ok(self.code.match_at(value, 0)?.map(|(_, last)| self.terminal(last)))
    }

    /// The terminal that owns group `last`. Only the winning alternative has groups set, so the
    /// highest set group is its outer group or one of its inner groups.
    fn terminal(&self, last: usize) -> &T {
        &self.payload[self.outer.partition_point(|&g| g <= last) - 1]
    }
}

fn pattern(flags: &str, alts: &str, whole: bool) -> String {
    let flags = if flags.is_empty() { String::new() } else { format!("(?{flags})") };
    let end = if whole { r"\z" } else { "" };
    format!(r"{flags}\G(?:{alts}){end}")
}

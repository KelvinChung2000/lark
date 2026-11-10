//! Owner types over the PCRE2 C API. The `pcre2` crate is not used because it exposes neither
//! the match limit nor a JIT stack shared between patterns.

use std::cell::RefCell;
use std::ffi::CString;
use std::ptr::{self, NonNull};

use pcre2_sys::*;

/// Upper bound of the per-thread JIT stack. The stack is reserved as virtual memory and
/// committed as a match uses it. A repeated group pushes a frame per iteration, so this bounds
/// the length of a token such as a run of whitespace under `(?:...)+`.
const JIT_STACK_MAX: usize = 1 << 30;

/// A pattern compiled in UTF and UCP mode, then JIT-compiled. It is read-only after
/// construction, so threads share it.
pub struct Code {
    ptr: NonNull<pcre2_code_8>,
    /// Capture groups plus the whole match, the ovector size a match needs.
    pairs: u32,
}

// SAFETY: PCRE2 documents compiled code as read-only during matching and safe to share.
unsafe impl Send for Code {}
unsafe impl Sync for Code {}

impl Code {
    pub fn new(pattern: &str) -> Result<Self, String> {
        let (mut err, mut offset) = (0, 0);
        // SAFETY: the pattern pointer and length describe a live UTF-8 buffer.
        let raw = unsafe {
            pcre2_compile_8(pattern.as_ptr(), pattern.len(), PCRE2_UTF | PCRE2_UCP, &mut err, &mut offset, ptr::null_mut())
        };
        let ptr = NonNull::new(raw).ok_or_else(|| format!("{} at offset {offset}", message(err)))?;
        let mut code = Self { ptr, pairs: 0 };
        let mut groups: u32 = 0;
        // SAFETY: `code.ptr` is a valid compiled pattern; CAPTURECOUNT writes a u32.
        let rc = unsafe { pcre2_pattern_info_8(code.ptr.as_ptr(), PCRE2_INFO_CAPTURECOUNT, (&mut groups as *mut u32).cast()) };
        assert_eq!(rc, 0, "PCRE2_INFO_CAPTURECOUNT failed");
        code.pairs = groups + 1;
        // SAFETY: as above.
        let rc = unsafe { pcre2_jit_compile_8(code.ptr.as_ptr(), PCRE2_JIT_COMPLETE) };
        if rc != 0 {
            return Err(format!("JIT compilation failed: {}", message(rc)));
        }
        Ok(code)
    }

    /// The number of the named group `name`.
    pub fn group(&self, name: &str) -> usize {
        let c = CString::new(name).expect("group names have no NUL");
        // SAFETY: valid code and a NUL-terminated name.
        let n = unsafe { pcre2_substring_number_from_name_8(self.ptr.as_ptr(), c.as_ptr().cast()) };
        assert!(n > 0, "the pattern has no group named {name}");
        n as usize
    }

    /// Matches at `start` of `subject`, which must be valid UTF-8 with `start` on a character
    /// boundary. Returns the match end and the highest capture group that took part.
    pub fn match_at(&self, subject: &str, start: usize) -> Result<Option<(usize, usize)>, String> {
        MATCHER.with_borrow_mut(|m| m.run(self, subject, start))
    }
}

impl Drop for Code {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from pcre2_compile_8 and is freed once.
        unsafe { pcre2_code_free_8(self.ptr.as_ptr()) }
    }
}

/// Match data, context and JIT stack of one thread. A match never calls back into Python, so
/// matches on a thread never nest and every pattern can share them.
struct Matcher {
    context: NonNull<pcre2_match_context_8>,
    stack: NonNull<pcre2_jit_stack_8>,
    data: NonNull<pcre2_match_data_8>,
    pairs: u32,
}

thread_local! {
    static MATCHER: RefCell<Matcher> = RefCell::new(Matcher::new());
}

impl Matcher {
    fn new() -> Self {
        // SAFETY: plain constructors with the default allocator; null results are checked.
        unsafe {
            let context = NonNull::new(pcre2_match_context_create_8(ptr::null_mut())).expect("pcre2_match_context_create failed");
            // Python's `re` has no step limit; the default of 10 million fails on long tokens.
            pcre2_set_match_limit_8(context.as_ptr(), u32::MAX);
            let stack = NonNull::new(pcre2_jit_stack_create_8(32 << 10, JIT_STACK_MAX, ptr::null_mut()))
                .expect("pcre2_jit_stack_create failed");
            pcre2_jit_stack_assign_8(context.as_ptr(), None, stack.as_ptr().cast());
            let data = NonNull::new(pcre2_match_data_create_8(1, ptr::null_mut())).expect("pcre2_match_data_create failed");
            Self { context, stack, data, pairs: 1 }
        }
    }

    fn run(&mut self, code: &Code, subject: &str, start: usize) -> Result<Option<(usize, usize)>, String> {
        if code.pairs > self.pairs {
            // SAFETY: the old match data is not borrowed; the new one is null-checked.
            unsafe {
                pcre2_match_data_free_8(self.data.as_ptr());
                self.data = NonNull::new(pcre2_match_data_create_8(code.pairs, ptr::null_mut())).expect("pcre2_match_data_create failed");
            }
            self.pairs = code.pairs;
        }
        // SAFETY: `code` is JIT-compiled with PCRE2_JIT_COMPLETE, `subject` is valid UTF-8 and
        // `start` is on a character boundary, which pcre2_jit_match requires instead of checking.
        let rc = unsafe {
            pcre2_jit_match_8(code.ptr.as_ptr(), subject.as_ptr(), subject.len(), start, 0, self.data.as_ptr(), self.context.as_ptr())
        };
        match rc {
            PCRE2_ERROR_NOMATCH => Ok(None),
            rc if rc < 0 => Err(message(rc)),
            0 => panic!("the ovector holds fewer pairs than the pattern has groups"),
            rc => {
                // SAFETY: the ovector holds `self.pairs` pairs and pair 0 is set after a match.
                let end = unsafe { *pcre2_get_ovector_pointer_8(self.data.as_ptr()).add(1) };
                Ok(Some((end, rc as usize - 1)))
            }
        }
    }
}

impl Drop for Matcher {
    fn drop(&mut self) {
        // SAFETY: each pointer was created in `new` or `run` and is freed once.
        unsafe {
            pcre2_match_data_free_8(self.data.as_ptr());
            pcre2_match_context_free_8(self.context.as_ptr());
            pcre2_jit_stack_free_8(self.stack.as_ptr());
        }
    }
}

fn message(code: i32) -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer and its length are valid; the result is the message length or negative.
    let n = unsafe { pcre2_get_error_message_8(code, buf.as_mut_ptr(), buf.len()) };
    match usize::try_from(n) {
        Ok(n) => String::from_utf8_lossy(&buf[..n]).into_owned(),
        Err(_) => format!("PCRE2 error {code}"),
    }
}

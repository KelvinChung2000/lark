fn main() {
    println!("cargo:rerun-if-env-changed=PCRE2_SYS_STATIC");
    if std::env::var("PCRE2_SYS_STATIC").as_deref() != Ok("1") {
        panic!("PCRE2_SYS_STATIC=1 is not set, so pcre2-sys may link a system libpcre2-8. Build from the lark-lalr-rs directory, whose .cargo/config.toml sets it, or export it.");
    }
}

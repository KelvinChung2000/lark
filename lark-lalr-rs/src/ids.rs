//! Dense indices into the native parse table. Each space is separate, so a state can never
//! index the symbol axis.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateId(pub u32);

/// A terminal or nonterminal name, interned once from the Python parse table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleId(pub u32);

impl StateId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl SymId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl RuleId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

pub fn id_u32(i: usize) -> u32 {
    u32::try_from(i).expect("parse table index exceeds u32")
}

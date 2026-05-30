#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageSelection {
    Auto,
    Named(String),
}

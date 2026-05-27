use burn_store::KeyRemapper;

const TALKER_LOAD_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.weight$", "${1}norm.gamma")];
const TALKER_EXPORT_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.gamma$", "${1}norm.weight")];

pub fn talker_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(TALKER_LOAD_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

pub fn talker_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(TALKER_EXPORT_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

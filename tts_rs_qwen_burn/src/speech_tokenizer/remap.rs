use burn_store::KeyRemapper;

const SPEECH_TOKENIZER_LOAD_KEY_PATTERNS: [(&str, &str); 1] = [(
    r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.weight$",
    "${1}.gamma",
)];
const SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS: [(&str, &str); 1] = [(
    r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.gamma$",
    "${1}.weight",
)];

pub(crate) fn speech_tokenizer_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(SPEECH_TOKENIZER_LOAD_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

pub(crate) fn speech_tokenizer_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

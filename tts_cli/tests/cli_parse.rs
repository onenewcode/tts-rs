use clap::Parser;
use tts_cli::Args;

#[test]
fn package_first_base_subcommand_parses_clone_flags() {
    let args = Args::try_parse_from([
        "tts_cli",
        "synthesize",
        "base",
        "--model-dir",
        "./Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "--text",
        "hello",
        "--ref-audio",
        "./clone.wav",
        "--ref-text",
        "reference words",
        "--output",
        "out.wav",
    ])
    .expect("package-first base command should parse");

    let debug = format!("{args:?}");
    assert!(debug.contains("Synthesize"));
    assert!(debug.contains("hello"));
    assert!(debug.contains("clone.wav"));
}

#[test]
fn package_first_custom_voice_subcommand_parses_speaker() {
    let args = Args::try_parse_from([
        "tts_cli",
        "synthesize",
        "custom-voice",
        "--manifest",
        "./custom/package.yaml",
        "--text",
        "hello",
        "--language",
        "zh",
        "--speaker",
        "Chelsie",
        "--output",
        "out.wav",
    ])
    .expect("package-first custom-voice command should parse");

    let debug = format!("{args:?}");
    assert!(debug.contains("Chelsie"));
    assert!(debug.contains("zh"));
}

#[test]
fn package_first_custom_voice_subcommand_parses_instruct() {
    let args = Args::try_parse_from([
        "tts_cli",
        "synthesize",
        "custom-voice",
        "--manifest",
        "./custom/package.yaml",
        "--text",
        "hello",
        "--speaker",
        "Vivian",
        "--instruct",
        "用特别开心的语气说",
        "--output",
        "out.wav",
    ])
    .expect("custom-voice instruct command should parse");

    let debug = format!("{args:?}");
    assert!(debug.contains("Vivian"));
    assert!(debug.contains("特别开心"));
}

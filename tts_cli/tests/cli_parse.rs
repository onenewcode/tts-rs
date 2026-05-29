use clap::Parser;
use tts_cli::Args;

#[test]
fn package_first_base_subcommand_parses() {
    let args = Args::try_parse_from([
        "tts_cli",
        "synthesize",
        "base",
        "--model-dir",
        "./Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "--text",
        "hello",
        "--output",
        "out.wav",
    ])
    .expect("package-first base command should parse");

    let debug = format!("{args:?}");
    assert!(debug.contains("Synthesize"));
    assert!(debug.contains("hello"));
}

#[test]
fn package_first_custom_voice_subcommand_parses_speaker_and_backend() {
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
        "--backend",
        "flex",
    ])
    .expect("package-first custom-voice command should parse");

    let debug = format!("{args:?}");
    assert!(debug.contains("Chelsie"));
    assert!(debug.contains("flex") || debug.contains("Flex"));
}

mod common;

fn main() {
    if let Err(error) = common::run_custom_voice_bf16() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

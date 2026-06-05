use clap::Parser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tts_cli::run(tts_cli::Args::parse())
}

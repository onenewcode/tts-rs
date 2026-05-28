#[inline]
pub fn should_emit_audio_chunk(pending_steps: usize, chunk_steps: usize, finished: bool) -> bool {
    pending_steps >= chunk_steps || (finished && pending_steps > 0)
}

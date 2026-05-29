use tts_infer::PcmAudio;

#[test]
fn write_wav_emits_pcm_header_for_audio_shape() {
    let audio = PcmAudio {
        pcm_i16: vec![1, -2, 3, -4],
        sample_rate: 24_000,
        channels: 2,
    };
    let mut bytes = Vec::new();

    audio
        .write_wav(&mut bytes)
        .expect("wav write should succeed");

    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1);
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 2);
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        24_000
    );
    assert_eq!(
        u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
        96_000
    );
    assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), 4);
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16);
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(
        u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
        8
    );
    assert_eq!(&bytes[44..], &[1, 0, 254, 255, 3, 0, 252, 255]);
}

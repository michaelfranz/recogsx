import librosa
import numpy as np


def __pad_input__(audio_input, target_len):
    # CNN needs input to be same length. This function truncates input if too long; pads input with zeros if troo short
    num_zeros_needed = target_len - len(audio_input)
    if num_zeros_needed > 0:
        num_zeros_front = np.random.randint(num_zeros_needed)
        num_zeros_back = num_zeros_needed - num_zeros_front
        return np.pad(audio_input, (num_zeros_front, num_zeros_back), mode='constant')
    else:
        return audio_input[0: target_len]


kpre_emphasis_coeff = 0.97


def __pre_emphasis__(audio_input):
    first_amp = audio_input[0]
    all_amps_without_first = audio_input[1:]
    all_amps_without_last = audio_input[:-1]
    emphasized_input = np.append(first_amp, all_amps_without_first - kpre_emphasis_coeff * all_amps_without_last)
    return emphasized_input


def audio_to_mfc(audio_input, sample_rate, target_len):
    padded_input = __pad_input__(audio_input, target_len)
    emphasized_input = __pre_emphasis__(padded_input)

    # apply dft, mel filter banks, logging, dct and normalization
    lifted_mfcc = librosa.feature.mfcc(
        y=emphasized_input.astype(float),
        sr=sample_rate,
        n_mfcc=12,
        dct_type=2,
        norm='ortho',
        lifter=22,
        n_fft=int(sample_rate * 0.025),
        hop_length=int(sample_rate * 0.01),
        power=2,
        center=False,
        window='hann',
        n_mels=40
    )

    return lifted_mfcc

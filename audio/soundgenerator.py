from preprocess import MinMaxNormaliser
import librosa

class SoundGenerator:
    """SoundGnerator is responsible for generating audio from spectrograms"""


    def __init__(self, vae, hop_length) -> None:
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrogram, min_max_values):
        generated_spectrogram, latent_representation = self.vae.reconstruct(spectrogram)

        signals = self.convert_spectrograms_to_audio(generated_spectrogram, min_max_values)

        return signals, latent_representation

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:,:,0]

            # apply denormalisation
            denrom_log_spec = self._min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])

            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denrom_log_spec)

            # apply Griffin-Lin
            signal = librosa.istft(spec, hop_length=self.hop_length)

            # append signal to "signals" List
            signals.append(signal)
        return signals
 
"""Impulse response container with metrics, WAV export, and plotting."""

import numpy as np
from scipy.io import wavfile


class ImpulseResponse:
    """Container for a room impulse response with ISO 3382 metrics."""

    def __init__(self, signal, fs=44100, label=''):
        self.signal = np.asarray(signal, dtype=np.float64)
        self.fs = fs
        self.label = label
        self._metrics = {}

    @property
    def t(self):
        return np.arange(len(self.signal)) / self.fs

    @property
    def duration(self):
        return len(self.signal) / self.fs

    # ── ISO 3382 metrics ─────────────────────────────────────

    @property
    def edc(self):
        """Energy decay curve (Schroeder integration)."""
        s2 = self.signal ** 2
        return np.cumsum(s2[::-1])[::-1]

    @property
    def edc_db(self):
        e = self.edc
        return 10 * np.log10(e / max(e[0], 1e-30) + 1e-30)

    def _find_decay_time(self, start_db, end_db):
        """Find reverberation time from EDC between two dB levels."""
        edc = self.edc_db
        t = self.t
        try:
            i_start = np.where(edc <= start_db)[0][0]
            i_end = np.where(edc <= end_db)[0][0]
            slope = (edc[i_end] - edc[i_start]) / (t[i_end] - t[i_start])
            return -60.0 / slope
        except (IndexError, ZeroDivisionError):
            return float('nan')

    @property
    def T30(self):
        """Reverberation time T30 [s]."""
        if 'T30' not in self._metrics:
            self._metrics['T30'] = self._find_decay_time(-5, -35)
        return self._metrics['T30']

    @property
    def T20(self):
        """Reverberation time T20 [s]."""
        if 'T20' not in self._metrics:
            self._metrics['T20'] = self._find_decay_time(-5, -25)
        return self._metrics['T20']

    @property
    def EDT(self):
        """Early decay time [s]."""
        if 'EDT' not in self._metrics:
            self._metrics['EDT'] = self._find_decay_time(0, -10)
        return self._metrics['EDT']

    @property
    def C80(self):
        """Clarity C80 [dB]."""
        if 'C80' not in self._metrics:
            n80 = int(0.08 * self.fs)
            s2 = self.signal ** 2
            early = np.sum(s2[:n80])
            late = np.sum(s2[n80:])
            self._metrics['C80'] = 10 * np.log10(
                early / max(late, 1e-30))
        return self._metrics['C80']

    @property
    def D50(self):
        """Definition D50."""
        if 'D50' not in self._metrics:
            n50 = int(0.05 * self.fs)
            s2 = self.signal ** 2
            self._metrics['D50'] = np.sum(s2[:n50]) / max(np.sum(s2), 1e-30)
        return self._metrics['D50']

    # ── Export ────────────────────────────────────────────────

    def to_wav(self, path, normalize=True):
        """Save as 16-bit WAV file."""
        s = self.signal
        if normalize:
            s = s / max(np.max(np.abs(s)), 1e-30) * 0.9
        wavfile.write(path, self.fs, (s * 32767).astype(np.int16))

    def to_npz(self, path):
        """Save raw data as numpy archive."""
        np.savez_compressed(path, signal=self.signal, fs=self.fs,
                            label=self.label)

    # ── Plotting ──────────────────────────────────────────────

    def plot(self, ax=None, show=True):
        """Plot waveform + EDC."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        else:
            axes = [ax, ax.twinx()]

        t_ms = self.t * 1000
        axes[0].plot(t_ms, self.signal, 'b-', lw=0.5)
        axes[0].set_ylabel('Pressure [Pa]')
        axes[0].set_title(self.label or 'Impulse Response')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_ms, self.edc_db, 'r-', lw=1)
        axes[1].set_ylabel('EDC [dB]')
        axes[1].set_xlabel('Time [ms]')
        axes[1].set_ylim(-60, 0)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return axes

    def plot_spectrogram(self, ax=None, show=True):
        """Plot spectrogram."""
        from scipy.signal import spectrogram
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        nperseg = min(256, len(self.signal) // 4)
        f, t, Sxx = spectrogram(self.signal, fs=self.fs,
                                 nperseg=nperseg,
                                 noverlap=nperseg * 3 // 4)
        Sxx_db = 10 * np.log10(Sxx + 1e-30)
        vmax = Sxx_db.max()
        vmin = max(vmax - 80, Sxx_db[Sxx_db > -300].min())
        ax.pcolormesh(t * 1000, f, Sxx_db, shading='gouraud',
                      cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [ms]')
        ax.set_ylim(0, min(self.fs // 2, 5000))
        ax.set_title(self.label or 'Spectrogram')
        plt.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self):
        return (f'ImpulseResponse({len(self.signal)} samples, '
                f'fs={self.fs}, T30={self.T30:.3f}s)')

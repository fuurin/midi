import numpy as np
import pretty_midi
from pretty_midi import PrettyMIDI, TimeSignature, KeySignature


"""
pypianorollとpretty_midiのresolutionは大きく違っていて，実際にデータ処理に使いたいresoutionはもっと小さかったりする．  
pretty_midiのresolution変えたりすると誤差がtempo_changesとかに生じるみたいなのでresolutionを変えずに時間⇔ポジション変換を行ったり，小節数を求めたりする関数を自分で定義
"""

def sec_to_pos(pm, sec, resolution=24):
    """
        second to position, on the given pretty_midi object
        ex. (array(0.0, 4.0,), array(120, 140,)) in the pm, 5.0 => 248
    """
    times, tempos = pm.get_tempo_changes()
    previous = times <= sec
    times, tempos = times[previous], tempos[previous]
    
    previous_steps = 0
    for i in range(0, len(times)-1):
        section_time = times[i + 1] - times[i]
        section_bps = tempos[i] / 60.0
        section_steps = section_time * section_bps * resolution
        previous_steps += section_steps

    last_time = sec - times[-1]
    last_bps = tempos[-1] / 60.0
    last_steps = last_time * last_bps * resolution

    total_steps = previous_steps + last_steps
    return int(np.floor(total_steps))


def pos_to_sec(pm, pos, resolution=24):
    """
        position to second, on the given pretty_midi object
        ex. (array(0.0, 4.0,), array(120, 140,)) in the pm, 248 => 5.0
    """
    times, tempos = pm.get_tempo_changes()
    
    previous_steps = 0
    previous_time = .0
    last_bps = tempos[0] / 60.0
    for i in range(0, len(times) - 1):
        section_time = times[i + 1] - times[i]
        section_steps = section_time * last_bps * resolution
        
        last_bps = tempos[i + 1] / 60.0
        if previous_steps + section_steps >= pos:
            break
        
        previous_time += section_time
        previous_steps += section_steps
    
    last_steps = pos - previous_steps
    last_time = last_steps / (last_bps * resolution)
    total_time = previous_time + last_time
    return total_time


def steps_to_bars(pm, steps, resolution=24):
    """
        returns which bar the given steps in on the given pretty_midi object
        ex. [('4/4', 0.0), ('3/4', 2.0), ('4/4', 3.5)] for bpm = 120 in the pm, 240 => 2
        note that the index of the first bar is 0
    """
    time_signatures = pm.time_signature_changes
    times = [ts.time for ts in time_signatures]
    numerators = [ts.numerator for ts in time_signatures]
    
    total_steps = 0
    total_bars = 0
    for i in range(1, len(times)):
        section_steps = sec_to_pos(pm, times[i], resolution) - total_steps
        
        if total_steps + section_steps > steps:
            break
        
        total_bars += section_steps / (resolution * numerators[i - 1])
        total_steps += section_steps
    
    last_steps = steps - total_steps
    total_bars += last_steps / (resolution * numerators[-1])
    return int(np.floor(total_bars))


def bars_to_steps(pm, bars, resolution=24):
    """
        returns absolute steps on the given pretty_midi object
    """
    time_signatures = pm.time_signature_changes
    times = [ts.time for ts in time_signatures]
    numerators = [ts.numerator for ts in time_signatures]
    
    total_steps = 0
    total_bars = 0
    for i in range(1, len(times)):
        section_steps = sec_to_pos(pm, times[i], resolution) - total_steps
        section_bars = section_steps / (resolution * numerators[i - 1])
        
        if total_bars + section_bars > bars:
            break
        
        total_bars += section_bars
        total_steps += section_steps
    
    last_bars = bars - total_bars
    total_steps += last_bars * resolution * numerators[-1]
    return int(np.floor(total_steps))



"""
与えられたMIDIデータのKeySignatureが信用できなかったり，そもそも書かれてなかったりすることがあるので，推定用の関数を用意
"""
def section(pm, start, end, index=None, exclude_drum=False):
    section_midi = PrettyMIDI()
    for i, instrument in enumerate(pm.instruments):
        if exclude_drum and instrument.is_drum: continue
        if index != None and i not in index: continue
        section_instrument = pretty_midi.Instrument(instrument.program)
        section_instrument.notes = [note for note in instrument.notes if start <= note.end < end]
        section_midi.instruments.append(section_instrument)
    
    return section_midi


def divide(pm, divider, index=None, exclude_drum=False):
    sections = []
    for i in range(1, len(divider)):
        sect = section(pm, divider[i-1], divider[i], index=index, exclude_drum=exclude_drum)
        sections.append(sect)
    return sections


def relative_chroma(pm, fs=100, vel_thresh=0):
    chroma = pm.get_chroma(fs=fs) > vel_thresh
    total_steps = sum(sum(chroma))
    if total_steps == 0:
        return None
    rel_chroma = np.array([sum(semitone)/total_steps for semitone in chroma]) # binarize
    return rel_chroma


def estimate_key_signature(pm, by_name=True, fs=100, vel_thresh=0):
    if len(pm.key_signature_changes) > 1:
        raise ValueError("estimate_key function is for single key signature midi data.")
    
    # 相対chromaを計算
    rel_chroma = relative_chroma(pm, fs=fs, vel_thresh=vel_thresh)
    if rel_chroma is None:
        return None, np.array([0] * 12)

    # 最もスケールに合致したシフト数を計算
    max_score, max_shift = 0.0, 0
    major_scale = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    for s in range(12):
        shift_scale = np.roll(major_scale, s).astype(bool)
        score = sum(rel_chroma[shift_scale])
        if score > max_score:
            max_score, max_shift = score, s
    
    # 1度が多ければMajor，6度が多ければminorとする
    if rel_chroma[max_shift] < rel_chroma[max_shift - 3]:
        key_number = (max_shift + -3) % 12 + 12
    else:
        key_number = max_shift
        
    return pretty_midi.key_number_to_key_name(key_number) if by_name else key_number, rel_chroma


def estimate_key_signature_changes(pm, index=None, by_name=True, fs=100, vel_thresh=0):
    ks_changes = [k for k in pm.key_signature_changes]
    ks_changes.append(KeySignature(ks_changes[-1].key_number, pm.get_end_time() + 1.0))
    
    estimated_keys = []
    for i in range(1, len(ks_changes)):
        section_start = ks_changes[i - 1].time
        section_end = ks_changes[i].time
        section_midi = section(pm, section_start, section_end, index=index, exclude_drum=True)
        key, _ = estimate_key_signature(section_midi, by_name=by_name, fs=fs, vel_thresh=vel_thresh)
        estimated_keys.append(key)
        
    return estimated_keys


"""
pypianorollのin_scale_rateが壊れているので自分で作成
"""
def in_scale_rate(pm, key_number, fs=100, vel_thresh=0):
    if key_number <= 11:
        scale = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    elif key_number <= 23:
        scale = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        key_number -= 12
    else:
        raise ValueError("Key number should be <= 23")

    scale = np.roll(scale, key_number).astype(bool)
    rel_chroma = relative_chroma(pm, fs=fs, vel_thresh=vel_thresh)
    scale_rate = round(sum(rel_chroma[scale]), 5)

    return scale_rate


def in_scale_rates(pm, key_changes=None, index=None, vel_thresh=0):
    key_changes = key_changes or pm.key_signature_changes or [KeySignature(0, 0.0)]
    key_changes = [kc for kc in key_changes]
    key_changes.append(KeySignature(key_changes[-1].key_number, pm.get_end_time() + 1.0))
    sections = divide(pm, [kc.time for kc in key_changes], index=index, exclude_drum=True)
    sect_keys = zip(sections, [kc.key_number for kc in key_changes])
    scale_rates = [in_scale_rate(sect, key) for sect, key in sect_keys]
    return scale_rates


if __name__ == "__main__":
    # Test
    resolution = 24
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    pm.time_signature_changes.append(TimeSignature(4,4,0.0))
    pm.time_signature_changes.append(TimeSignature(3,4,(60 / 120) * 4))
    pm.time_signature_changes.append(TimeSignature(4,4,(60 / 120) * 7))
    print([(f"{ts.numerator}/{ts.denominator}", ts.time) for ts in pm.time_signature_changes])

    print("head of bar[1]:", steps_to_bars(pm, resolution * 4))
    print("one step before head of bar[1]:", steps_to_bars(pm, resolution * 4 - 1))
    print("head of bar[2]:", steps_to_bars(pm, resolution * 7))
    print("one step before head of bar[2]:", steps_to_bars(pm, resolution * 7 - 1))
    print("head of bar[3]:", steps_to_bars(pm, resolution * 11))
    print("one step before head of bar[3]:", steps_to_bars(pm, resolution * 11 - 1))
    print("bar id of 240 steps:", steps_to_bars(pm, 240))

    print("\nbars_to_steps => actual steps")
    print(bars_to_steps(pm, 1), resolution * 4)
    print(bars_to_steps(pm, 2), resolution * (4 + 3))
    print(bars_to_steps(pm, 3), resolution * (4 + 3 + 4))
    print(bars_to_steps(pm, 4), resolution * (4 + 3 + 4 * 2))
    
    
    midi_path = "../datasets/theorytab/pianoroll/a/a-ha/take-on-me/intro-and-verse_key.mid"
    pm = PrettyMIDI(midi_path)
    del pm.instruments[1]
    pm.key_signature_changes += [KeySignature(8, 14.0)]
    print("\nDivided take-on-me key estimation")
    print(estimate_key_signature_changes(pm))
    
    scale_rates = in_scale_rates(pm)
    print("\nscale rates")
    print(scale_rates)
    
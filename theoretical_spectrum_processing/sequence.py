from .constants import MOD_MASS_TO_CODE

class SequenceProcessor:
    @staticmethod
    def get_decoy(pep):
        clean_seq, modifications = SequenceProcessor.parse_modifications(pep)
        
        if len(clean_seq) <= 2:
            return pep
        
        middle = clean_seq[1:-1]
        decoy_clean = clean_seq[0] + middle[::-1] + clean_seq[-1]
        
        decoy_mods = []
        for residue, pos, mass_shift in modifications:
            if pos == -1:
                decoy_mods.append((residue, pos, mass_shift))
            else:
                new_pos = SequenceProcessor.map_decoy_position(pos, len(clean_seq))
                decoy_mods.append((residue, new_pos, mass_shift))
        
        decoy_with_mods = SequenceProcessor.reconstruct_sequence_with_modifications(decoy_clean, decoy_mods)
        return decoy_with_mods
    
    @staticmethod
    def map_decoy_position(pos, seq_len):
        if seq_len <= 2:
            return pos
            
        if pos == 0 or pos == seq_len - 1:
            return pos
        
        return seq_len - 1 - pos
    
    @staticmethod
    def reconstruct_sequence_with_modifications(clean_seq, modifications):
        mass_to_letter = dict(MOD_MASS_TO_CODE)

        result = ""

        nterm_mods = [mod for mod in modifications if mod[1] == -1]
        for _, _, mass_shift in nterm_mods:
            key = f"{mass_shift:.3f}"
            result += mass_to_letter[key]

        for i, residue in enumerate(clean_seq):
            result += residue
            residue_mods = [mod for mod in modifications if mod[1] == i]
            for _, _, mass_shift in residue_mods:
                key = f"{mass_shift:.3f}"
                result += mass_to_letter[key]
        return result
    
    @staticmethod
    def parse_modifications(peptide_with_mods):
        letter_to_mass = {letter: float(mass) for mass, letter in MOD_MASS_TO_CODE.items()}

        modifications = []
        clean_chars = []
        last_res_pos = -1

        for ch in peptide_with_mods:
            if 'A' <= ch <= 'Z':
                clean_chars.append(ch)
                last_res_pos += 1
            elif 'a' <= ch <= 'z':
                if ch in letter_to_mass:
                    mass_shift = letter_to_mass[ch]
                    if last_res_pos == -1:
                        modifications.append((None, -1, mass_shift))
                    else:
                        residue = clean_chars[last_res_pos]
                        modifications.append((residue, last_res_pos, mass_shift))
                else:
                    raise ValueError(f"Unknown modification code: {ch}")
            else:
                raise ValueError(f"Invalid peptide character: {ch}")

        clean_sequence = ''.join(clean_chars)
        return clean_sequence, modifications

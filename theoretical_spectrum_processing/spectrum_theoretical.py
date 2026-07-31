from .constants import AA_MONO_MASSES, PROTON_MASS, WATER_MASS
from .sequence import SequenceProcessor

class SpectrumTheoreticalProcessor:
    @staticmethod
    def _fragment_mz(peptide, ion_type, charge, modifications):
        mass = sum(AA_MONO_MASSES[aa] for aa in peptide)

        for _, pos, mass_shift in modifications:
            if pos == -1 or 0 <= pos < len(peptide):
                mass += mass_shift

        if ion_type.startswith("y"):
            mass += WATER_MASS

        return (mass + charge * PROTON_MASS) / charge

    @staticmethod
    def _fragments(peptide, maxcharge):
        clean_peptide, modifications = SequenceProcessor.parse_modifications(peptide)

        for i in range(1, len(clean_peptide)):
            for ion_type in ("b", "y"):
                for charge in range(1, maxcharge + 1):
                    if ion_type == "b":
                        fragment_seq = clean_peptide[:i]
                        fragment_mods = [
                            mod for mod in modifications
                            if mod[1] == -1 or 0 <= mod[1] < i
                        ]
                    else:
                        fragment_seq = clean_peptide[i:]
                        fragment_mods = [
                            (residue, pos - i, mass_shift)
                            for residue, pos, mass_shift in modifications
                            if pos >= i
                        ]

                    mz = SpectrumTheoreticalProcessor._fragment_mz(
                        fragment_seq, ion_type, charge, fragment_mods
                    )
                    yield mz, ion_type, charge

    @staticmethod
    def generate_theoretical_spectrum(
        peptide,
        meta,
        maxcharge=2
    ):
        charge = meta["charge"]
        if isinstance(charge, int) and charge > 0:
            maxcharge = int(charge)

        meta = dict(meta)
        full_pep, full_mods = SequenceProcessor.parse_modifications(peptide)
        prec_mass = sum(AA_MONO_MASSES[aa] for aa in full_pep)
        prec_mass += sum(mass_shift for _, _, mass_shift in full_mods)
        precursor_charge = max(maxcharge, 1)
        meta["precursor_mz"] = float((prec_mass + WATER_MASS + precursor_charge * PROTON_MASS) / precursor_charge)
        meta["precursor_charge"] = precursor_charge
        meta["sequence"] = peptide

        frags = list(SpectrumTheoreticalProcessor._fragments(peptide, maxcharge))
        if not frags:
            return [], meta

        weights = [
            {"y": 1.0, "b": 0.85}[ion] / (charge ** 0.8)
            for _, ion, charge in frags
        ]

        max_weight = max(weights) if weights else 0.0
        if max_weight > 0:
            weights = [weight / max_weight for weight in weights]
        else:
            weights = [0.0 for _ in weights]

        merged = {}
        for (mz, _, _), weight in zip(frags, weights):
            intensity = min(max(round(weight * 255.0), 0), 255)
            if mz not in merged or intensity > merged[mz]:
                merged[mz] = int(intensity)

        return sorted(merged.items(), key=lambda item: item[0]), meta

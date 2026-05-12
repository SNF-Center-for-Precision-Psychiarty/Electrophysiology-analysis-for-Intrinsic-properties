import warnings, os
warnings.filterwarnings("ignore")
from pynwb import NWBHDF5IO
import numpy as np

paths = [
    r"K:\2. Electrophysiology\Analysis_Subset\sub-1058766119\sub-1058766119_ses-1059241097_icephys.nwb",
    r"K:\2. Electrophysiology\Analysis_Subset\sub-1058766119\sub-1058766119_ses-1059245935_icephys.nwb",
    r"K:\2. Electrophysiology\Analysis_Subset\sub-1058766119\sub-1058766119_ses-1060105395_icephys.nwb",
    r"K:\2. Electrophysiology\sub-210210\sub-210210_ses-2102101si-2-1-1_icephys.nwb",
    r"K:\2. Electrophysiology\sub-210210\sub-210210_ses-2102101tm-4-1-1_icephys.nwb",
]

for p in paths:
    try:
        with NWBHDF5IO(p, "r") as io:
            nwb = io.read()
            stim_keys = list(nwb.stimulus.keys())
            acq_keys = list(nwb.acquisition.keys())

            tot_data = 0
            tot_nan = 0
            sw_with_nan = 0
            sw_nontrailing = 0
            sw_all_nan = 0
            unit_set = set()
            buf_lens = set()

            for k in stim_keys:
                ts = nwb.stimulus[k]
                d = np.asarray(ts.data)
                unit_set.add(getattr(ts, "unit", "?"))
                buf_lens.add(len(d))
                n = int(np.isnan(d).sum())
                tot_data += len(d)
                tot_nan += n
                if n == 0:
                    continue
                sw_with_nan += 1
                mask = np.isnan(d)
                if mask.all():
                    sw_all_nan += 1
                    continue
                first_nan = int(np.argmax(mask))
                if (not mask[first_nan:].all()) or mask[:first_nan].any():
                    sw_nontrailing += 1
            for k in acq_keys:
                ts = nwb.acquisition[k]
                d = np.asarray(ts.data)
                unit_set.add(getattr(ts, "unit", "?"))
                buf_lens.add(len(d))
                n = int(np.isnan(d).sum())
                tot_data += len(d)
                tot_nan += n
                if n == 0:
                    continue
                sw_with_nan += 1
                mask = np.isnan(d)
                if mask.all():
                    sw_all_nan += 1
                    continue
                first_nan = int(np.argmax(mask))
                if (not mask[first_nan:].all()) or mask[:first_nan].any():
                    sw_nontrailing += 1

            short = os.path.basename(p)
            pct = 100 * tot_nan / tot_data if tot_data else 0
            print(short)
            print(f"  stim+acq=({len(stim_keys)}+{len(acq_keys)})  units={unit_set}  buf_lens={sorted(buf_lens)[:6]}{'...' if len(buf_lens)>6 else ''}")
            print(f"  total={tot_data:,}  NaN={tot_nan:,} ({pct:.2f}%)  with_NaN={sw_with_nan}  non_trailing={sw_nontrailing}  all_NaN={sw_all_nan}")
    except Exception as e:
        print(f"{p}: ERROR {e}")

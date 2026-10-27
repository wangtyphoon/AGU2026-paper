"""The source USA-only subset must load without invented WMO fields."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from agu_james.plotting.krathon_track import KNOT_TO_MPS, load_ibtracs


class KrathonInput(unittest.TestCase):
    def test_usa_only_subset_and_source_provenance(self):
        scratch = Path(__file__).resolve().parents[1] / 'artifacts/verification'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source = Path(directory) / 'subset.csv'
            source.write_text('SID,SEASON,NAME,ISO_TIME,TRACK_TYPE,USA_LAT,USA_LON,USA_WIND,USA_PRES,original_source_csv\n'
                'storm,2024,KRATHON,2024-09-30 12:00:00,main,20,125,100,950,upstream.csv\n', encoding='utf-8')
            paired = pd.DataFrame({'valid_time_utc': pd.to_datetime(['2024-09-30 12:00:00'], utc=True)})
            frame = load_ibtracs(paired, source)
            self.assertEqual(frame.USA_LAT.iloc[0], 20)
            self.assertEqual(frame.USA_WIND.iloc[0], 100)
            self.assertEqual(frame.USA_wind_mps.iloc[0], 100 * KNOT_TO_MPS)
            self.assertEqual(frame.original_source_csv.iloc[0], 'upstream.csv')
            self.assertEqual(frame.source_csv.iloc[0], str(source.resolve()))
            self.assertNotIn('WMO_WIND', frame)
            source.write_text(source.read_text().replace('USA_LAT', 'LAT'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'USA_LAT'):
                load_ibtracs(paired, source)


if __name__ == '__main__':
    unittest.main()

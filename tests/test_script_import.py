import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.getenv('YEGOU_GATE_TEST') == '1', 'requires isolated workbench environment')
class ScriptTests(unittest.TestCase):
    def setUp(self):
        from app import dubbing
        self.mod=dubbing
        self.id='bbbbbbbbbbbb'
        self.root=dubbing.DUBBING_WORK/self.id
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'timeline.json'
        self.path.write_text(json.dumps({'duration_sec':10,'segments':[{'text':'old'}],'speakers':[]}))
        self.mod.JOBS[self.id]={'job_id':self.id,'status':'done','timeline_path':str(self.path),'output_path':str(self.root)}
        self.speakers=[{'speaker_id':'a','speaker_label':'A'}]
        self.cues=[{'speaker_id':'a','start_sec':1,'end_sec':2,'text':'你好'}]

    def tearDown(self):
        self.mod.JOBS.pop(self.id,None)
        for p in self.root.iterdir():
            p.unlink()
        self.root.rmdir()

    def test_script_replaces_bad_asr_preserves_original_and_sets_ids(self):
        result=self.mod.import_script(self.id,self.speakers,self.cues)
        timeline=result['timeline']
        self.assertEqual(timeline['segments'][0]['segment_id'],'seg_0001')
        self.assertEqual(timeline['original_analysis']['segments'][0]['text'],'old')
        self.assertEqual(timeline['segments'][0]['dubbing_text'],'你好')

    def test_reject_unknown_role_overrun_overlap_and_duplicate_speakers(self):
        for speakers,cues in [
            (self.speakers,[dict(self.cues[0],speaker_id='other')]),
            (self.speakers,[dict(self.cues[0],end_sec=12)]),
            (self.speakers,self.cues+self.cues),
            (self.speakers*2,self.cues),
        ]:
            with self.assertRaises(ValueError):
                self.mod.import_script(self.id,speakers,cues)
        self.assertEqual(json.loads(self.path.read_text())['segments'][0]['text'],'old')

    def test_cannot_replace_running_task(self):
        self.mod.JOBS[self.id]['status']='generating'
        with self.assertRaises(ValueError):
            self.mod.import_script(self.id,self.speakers,self.cues)


class TimestampTests(unittest.TestCase):
    def test_subtitle_millisecond_rounding(self):
        from app.agent_media import stamp
        self.assertEqual(stamp(59.9999),'00:01:00,000')
        self.assertEqual(stamp(3661.234),'01:01:01,234')

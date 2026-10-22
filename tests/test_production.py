import copy
import unittest
from app.production import PROFILE, validate_job


class ProductionContractTests(unittest.TestCase):
    def job(self):
        return {'workspace': '/project/agent_work', 'input': '/project/video.mp4',
                'references': {'a': {'audio': '/project/a.wav', 'text': '参考台词'}},
                'segments': [{'speaker_id': 'a', 'start_sec': 1, 'end_sec': 2, 'text': '你好',
                              'tts_text': '[excited]你好'}]}

    def test_accepted_profile_keeps_demucs_background_and_fish(self):
        self.assertEqual(PROFILE['separation_model'], 'HDEMUCS_HIGH_MUSDB')
        self.assertEqual(PROFILE['background_sources'], ['drums', 'bass', 'other'])
        self.assertEqual(PROFILE['tts_model'], 's2-pro-bnb-nf4')
        self.assertEqual(PROFILE['background_volume'], 1)
        self.assertEqual(validate_job(self.job(), 'run')['segments'][0]['text'], '你好')

    def test_rejects_bandit_and_unlabelled_cached_background(self):
        job = self.job()
        job['separation_model'] = 'bandit_v2'
        with self.assertRaisesRegex(ValueError, 'HDEMUCS'):
            validate_job(job, 'run')
        job.pop('separation_model')
        job['background_audio'] = '/cache/background.wav'
        with self.assertRaisesRegex(ValueError, 'background_method'):
            validate_job(job, 'run')
        job['background_method'] = 'HDEMUCS_HIGH_MUSDB'
        validate_job(job, 'run')

    def test_rejects_overlap_nan_empty_and_missing_reference(self):
        for patch in [{'start_sec': float('nan')}, {'end_sec': float('inf')}, {'text': ''}, {'speaker_id': 'missing'}]:
            job = self.job()
            job['segments'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_job(job, 'run')
        job = self.job()
        job['segments'].append(dict(job['segments'][0], start_sec=1.5, end_sec=3))
        with self.assertRaises(ValueError):
            validate_job(job, 'run')

    def test_render_requires_clips_and_original_background(self):
        job = self.job()
        with self.assertRaisesRegex(ValueError, 'audio_path'):
            validate_job(job, 'render')
        job['segments'][0]['audio_path'] = '/cache/generated.wav'
        with self.assertRaisesRegex(ValueError, 'background_audio'):
            validate_job(job, 'render')
        job.update(background_audio='/cache/original-demucs.wav', background_method='HDEMUCS_HIGH_MUSDB')
        job.pop('references')
        validate_job(job, 'render')

    def test_rejects_unsafe_synthesis_and_mix_ranges(self):
        for key, value in [('temperature', float('nan')), ('top_p', 4), ('gain', -2), ('max_new_tokens', 0)]:
            job = self.job()
            job['segments'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_job(job, 'tts')
        job = self.job()
        job['background_volume'] = -1
        with self.assertRaises(ValueError):
            validate_job(job, 'run')


if __name__ == '__main__':
    unittest.main()

import unittest

from app.services.documentary.frame_analysis_models import DocumentaryAnalysisConfig
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService


class DocumentaryFrameAnalysisServiceTests(unittest.TestCase):
    def test_build_analysis_prompt_formats_real_frame_count(self):
        service = DocumentaryFrameAnalysisService()

        prompt = service._build_analysis_prompt(frame_count=3)

        self.assertIn("我提供了 3 张视频帧", prompt)
        self.assertNotIn("%s", prompt)
        self.assertIn("frame_observations", prompt)
        self.assertIn("overall_activity_summary", prompt)

    def test_parse_failed_batch_keeps_raw_response_and_time_range(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._build_failed_batch_result(
            batch_index=2,
            raw_response="not-json",
            error_message="JSON decode failed",
            frame_paths=["/tmp/keyframe_000000_000000000.jpg"],
            time_range="00:00:00,000-00:00:03,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("not-json", batch.raw_response)
        self.assertEqual("00:00:00,000-00:00:03,000", batch.time_range)
        self.assertTrue(batch.fallback_summary)

    def test_parse_failed_batch_uses_non_empty_fallback_when_raw_response_is_empty(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._build_failed_batch_result(
            batch_index=3,
            raw_response="",
            error_message="Empty model response",
            frame_paths=["/tmp/keyframe_000001_000001000.jpg"],
            time_range="00:00:03,000-00:00:06,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("", batch.raw_response)
        self.assertTrue(batch.fallback_summary)


class DocumentaryAnalysisConfigTests(unittest.TestCase):
    def test_config_rejects_non_positive_frame_interval(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=0,
                vision_batch_size=5,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
            )

    def test_config_rejects_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=5,
                vision_batch_size=0,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
            )

    def test_config_rejects_non_positive_max_concurrency(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=5,
                vision_batch_size=5,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
                max_concurrency=0,
            )

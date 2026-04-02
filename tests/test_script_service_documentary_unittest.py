import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.services.script_service import ScriptGenerator


class ScriptGeneratorDocumentaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_script_forwards_explicit_values_to_shared_service(self):
        expected_script = [
            {
                "timestamp": "00:00:00,000-00:00:03,000",
                "picture": "批次描述",
                "narration": "这里是解说词",
                "OST": 2,
            }
        ]
        callback = lambda _percent, _message: None

        with patch("app.services.script_service.DocumentaryFrameAnalysisService") as service_cls:
            service = service_cls.return_value
            service.generate_documentary_script = AsyncMock(return_value=expected_script)
            generator = ScriptGenerator()

            result = await generator.generate_script(
                video_path="demo.mp4",
                video_theme="荒野生存",
                custom_prompt="请聚焦生存动作",
                frame_interval_input=3,
                vision_batch_size=6,
                vision_llm_provider="openai",
                progress_callback=callback,
            )

        self.assertEqual(expected_script, result)
        self.assertTrue(result[0]["narration"])
        service.generate_documentary_script.assert_awaited_once()
        called_kwargs = service.generate_documentary_script.await_args.kwargs
        self.assertEqual("demo.mp4", called_kwargs["video_path"])
        self.assertEqual(3, called_kwargs["frame_interval_input"])
        self.assertEqual(6, called_kwargs["vision_batch_size"])
        self.assertEqual("openai", called_kwargs["vision_llm_provider"])
        self.assertEqual("荒野生存", called_kwargs["video_theme"])
        self.assertEqual("请聚焦生存动作", called_kwargs["custom_prompt"])
        self.assertIs(called_kwargs["progress_callback"], callback)

    async def test_generate_script_forwards_unset_values_as_none(self):
        expected_script = [
            {
                "timestamp": "00:00:00,000-00:00:03,000",
                "picture": "批次描述",
                "narration": "这里是解说词",
                "OST": 2,
            }
        ]
        with patch("app.services.script_service.DocumentaryFrameAnalysisService") as service_cls:
            service = service_cls.return_value
            service.generate_documentary_script = AsyncMock(return_value=expected_script)
            generator = ScriptGenerator()

            await generator.generate_script(video_path="demo.mp4")

        called_kwargs = service.generate_documentary_script.await_args.kwargs
        self.assertIsNone(called_kwargs["frame_interval_input"])
        self.assertIsNone(called_kwargs["vision_batch_size"])
        self.assertIsNone(called_kwargs["vision_llm_provider"])

    async def test_generate_script_warns_when_skip_seconds_or_threshold_are_non_default(self):
        expected_script = [
            {
                "timestamp": "00:00:00,000-00:00:03,000",
                "picture": "批次描述",
                "narration": "这里是解说词",
                "OST": 2,
            }
        ]
        with patch("app.services.script_service.DocumentaryFrameAnalysisService") as service_cls, patch(
            "app.services.script_service.logger.warning"
        ) as warning:
            service = service_cls.return_value
            service.generate_documentary_script = AsyncMock(return_value=expected_script)
            generator = ScriptGenerator()
            await generator.generate_script(
                video_path="demo.mp4",
                skip_seconds=2,
                threshold=20,
            )

        warning.assert_called_once()
        warning_message = warning.call_args.args[0]
        self.assertIn("skip_seconds", warning_message)
        self.assertIn("threshold", warning_message)
        self.assertIn("does not currently apply", warning_message)


class DocumentaryFrameAnalysisServiceScriptGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_documentary_script_returns_final_narrated_items(self):
        service = DocumentaryFrameAnalysisService()
        analysis_payload = {
            "batches": [
                {
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:03,000",
                    "overall_activity_summary": "",
                    "fallback_summary": "回退摘要",
                    "frame_observations": [
                        {"timestamp": "00:00:00,000", "observation": "镜头里有一只猫"},
                    ],
                }
            ]
        }

        with TemporaryDirectory() as temp_dir:
            analysis_path = Path(temp_dir) / "frame_analysis_test.json"
            analysis_path.write_text(json.dumps(analysis_payload, ensure_ascii=False), encoding="utf-8")

            with patch.object(
                DocumentaryFrameAnalysisService,
                "analyze_video",
                AsyncMock(return_value={"analysis_json_path": str(analysis_path)}),
            ), patch.dict(
                "app.services.documentary.frame_analysis_service.config.app",
                {
                    "text_llm_provider": "openai",
                    "text_openai_api_key": "test-key",
                    "text_openai_model_name": "test-model",
                    "text_openai_base_url": "https://example.com/v1",
                },
            ), patch(
                "app.services.documentary.frame_analysis_service.generate_narration",
                return_value='{"items":[{"timestamp":"00:00:00,000-00:00:03,000","picture":"镜头里有一只猫","narration":"一只猫警觉地望向镜头。"}]}',
            ):
                result = await service.generate_documentary_script(video_path="demo.mp4")

        self.assertEqual(1, len(result))
        self.assertEqual("00:00:00,000-00:00:03,000", result[0]["timestamp"])
        self.assertEqual("镜头里有一只猫", result[0]["picture"])
        self.assertEqual("一只猫警觉地望向镜头。", result[0]["narration"])
        self.assertEqual(2, result[0]["OST"])


if __name__ == "__main__":
    unittest.main()

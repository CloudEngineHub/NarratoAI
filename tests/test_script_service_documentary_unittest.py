import unittest
from unittest.mock import AsyncMock, patch

from app.services.script_service import ScriptGenerator


class ScriptGeneratorDocumentaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_script_passes_frame_interval_to_shared_service(self):
        expected_script = [
            {
                "timestamp": "00:00:00,000-00:00:03,000",
                "picture": "批次描述",
                "narration": "",
                "OST": 2,
            }
        ]
        progress = []

        def progress_callback(percent, message):
            progress.append((percent, message))

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
                progress_callback=progress_callback,
            )

        self.assertEqual(expected_script, result)
        service.generate_documentary_script.assert_awaited_once()
        called_kwargs = service.generate_documentary_script.await_args.kwargs
        self.assertEqual("demo.mp4", called_kwargs["video_path"])
        self.assertEqual(3, called_kwargs["frame_interval_input"])
        self.assertEqual(6, called_kwargs["vision_batch_size"])
        self.assertEqual("openai", called_kwargs["vision_llm_provider"])
        self.assertEqual("荒野生存", called_kwargs["video_theme"])
        self.assertEqual("请聚焦生存动作", called_kwargs["custom_prompt"])
        self.assertIs(called_kwargs["progress_callback"], progress_callback)
        self.assertEqual([], progress)


if __name__ == "__main__":
    unittest.main()

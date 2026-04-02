"""OpenAI 兼容 provider 的最小回归测试。"""

import asyncio
import unittest

from app.config import config
from app.services.llm.base import TextModelProvider
from app.services.llm.manager import LLMServiceManager
from app.services.llm.openai_compatible_provider import OpenAICompatibleVisionProvider
from app.services.llm.providers import register_all_providers


class DummyOpenAITextProvider(TextModelProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return prompt

    async def _make_api_call(self, payload: dict) -> dict:
        return payload


def _reset_manager_state():
    LLMServiceManager._vision_providers.clear()
    LLMServiceManager._text_providers.clear()
    LLMServiceManager._vision_instance_cache.clear()
    LLMServiceManager._text_instance_cache.clear()


class OpenAICompatManagerTests(unittest.TestCase):
    def setUp(self):
        _reset_manager_state()
        self._original_app = dict(config.app)

    def tearDown(self):
        _reset_manager_state()
        config.app.clear()
        config.app.update(self._original_app)

    def test_register_all_providers_only_registers_openai_provider(self):
        register_all_providers()

        self.assertEqual({"openai"}, set(LLMServiceManager.list_text_providers()))
        self.assertEqual({"openai"}, set(LLMServiceManager.list_vision_providers()))

    def test_get_text_provider_uses_openai_keys(self):
        LLMServiceManager.register_text_provider("openai", DummyOpenAITextProvider)

        config.app["text_llm_provider"] = "openai"
        config.app["text_openai_api_key"] = "new-key"
        config.app["text_openai_model_name"] = "new-model"
        config.app["text_openai_base_url"] = "https://new.example/v1"

        provider = LLMServiceManager.get_text_provider()

        self.assertIsInstance(provider, DummyOpenAITextProvider)
        self.assertEqual("new-key", provider.api_key)
        self.assertEqual("new-model", provider.model_name)
        self.assertEqual("https://new.example/v1", provider.base_url)


class OpenAICompatVisionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_images_keeps_batch_order_when_running_concurrently(self):
        provider = OpenAICompatibleVisionProvider(api_key="k", model_name="m")
        provider._prepare_images = lambda images: list(images)

        async def fake_analyze_batch(batch, prompt, **kwargs):
            delays = {"a": 0.03, "c": 0.01, "e": 0.0}
            await asyncio.sleep(delays[batch[0]])
            return f"batch-{batch[0]}"

        provider._analyze_batch = fake_analyze_batch

        result = await provider.analyze_images(
            images=["a", "b", "c", "d", "e", "f"],
            prompt="prompt",
            batch_size=2,
            max_concurrency=2,
        )

        self.assertEqual(["batch-a", "batch-c", "batch-e"], result)

    async def test_analyze_images_respects_max_concurrency_limit(self):
        provider = OpenAICompatibleVisionProvider(api_key="k", model_name="m")
        provider._prepare_images = lambda images: list(images)

        in_flight = 0
        max_in_flight = 0

        async def fake_analyze_batch(batch, prompt, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"batch-{batch[0]}"

        provider._analyze_batch = fake_analyze_batch

        result = await provider.analyze_images(
            images=["a", "b", "c", "d", "e", "f"],
            prompt="prompt",
            batch_size=1,
            max_concurrency=2,
        )

        self.assertEqual(6, len(result))
        self.assertEqual(2, max_in_flight)


if __name__ == "__main__":
    unittest.main()

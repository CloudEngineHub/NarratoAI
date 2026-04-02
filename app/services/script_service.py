from typing import Any, Callable

from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService


class ScriptGenerator:
    def __init__(self, documentary_service: DocumentaryFrameAnalysisService | None = None):
        self.documentary_service = documentary_service or DocumentaryFrameAnalysisService()

    async def generate_script(
        self,
        video_path: str,
        video_theme: str = "",
        custom_prompt: str = "",
        frame_interval_input: int = 5,
        skip_seconds: int = 0,
        threshold: int = 30,
        vision_batch_size: int = 5,
        vision_llm_provider: str = "gemini",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[dict[Any, Any]]:
        callback = progress_callback or (lambda _p, _m: None)
        return await self.documentary_service.generate_documentary_script(
            video_path=video_path,
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            frame_interval_input=frame_interval_input,
            vision_batch_size=vision_batch_size,
            vision_llm_provider=vision_llm_provider,
            progress_callback=callback,
            # 历史参数保留在签名中以兼容调用方；共享逐帧分析当前不使用这两个参数。
            # skip_seconds=skip_seconds,
            # threshold=threshold,
        )

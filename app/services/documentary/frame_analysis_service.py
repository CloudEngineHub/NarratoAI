from app.services.documentary.frame_analysis_models import FrameBatchResult


class DocumentaryFrameAnalysisService:
    PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧，它们按时间顺序排列，代表一个连续的视频片段。
首先，请详细描述每一帧的关键视觉信息（包含：主要内容、人物、动作和场景）。
然后，基于所有帧的分析，请用简洁的语言总结整个视频片段中发生的主要活动或事件流程。
请务必使用 JSON 格式输出。
请务必不要遗漏视频帧，我提供了 {frame_count} 张视频帧，frame_observations 必须包含 {frame_count} 个元素
""".strip()

    def _build_analysis_prompt(self, frame_count: int) -> str:
        return self.PROMPT_TEMPLATE.format(frame_count=frame_count)

    def _build_failed_batch_result(
        self,
        *,
        batch_index: int,
        raw_response: str,
        error_message: str,
        frame_paths: list[str],
        time_range: str,
    ) -> FrameBatchResult:
        fallback_summary = (raw_response or "").strip()[:200]
        if not fallback_summary:
            fallback_summary = f"Batch {batch_index} analysis failed: {error_message or 'unknown error'}"

        return FrameBatchResult(
            batch_index=batch_index,
            status="failed",
            time_range=time_range,
            raw_response=raw_response,
            frame_paths=list(frame_paths),
            fallback_summary=fallback_summary,
            error_message=error_message,
        )

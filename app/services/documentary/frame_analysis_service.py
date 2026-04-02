import json
import os
import re

from app.utils import utils
from app.services.documentary.frame_analysis_models import FrameBatchResult


class DocumentaryFrameAnalysisService:
    PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧，它们按时间顺序排列，代表一个连续的视频片段。
首先，请详细描述每一帧的关键视觉信息（包含：主要内容、人物、动作和场景）。
然后，基于所有帧的分析，请用简洁的语言总结整个视频片段中发生的主要活动或事件流程。
请务必使用 JSON 格式输出。
JSON 必须包含以下键：
- frame_observations: 数组，且长度必须为 {frame_count}
- overall_activity_summary: 字符串，描述整个批次主要活动
示例结构：
{{
  "frame_observations": [
    {{"timestamp": "00:00:00,000", "observation": "画面描述"}}
  ],
  "overall_activity_summary": "本批次主要活动总结"
}}
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

    def _build_cache_key(
        self,
        video_path: str,
        interval_seconds: float,
        prompt_version: str,
        model_name: str,
        batch_size: int,
        max_concurrency: int,
    ) -> str:
        try:
            video_mtime = os.path.getmtime(video_path)
        except OSError:
            video_mtime = 0

        legacy_prefix = utils.md5(f"{video_path}{video_mtime}")

        payload = "|".join(
            [
                str(video_path),
                str(video_mtime),
                str(interval_seconds),
                str(prompt_version),
                str(model_name),
                str(batch_size),
                str(max_concurrency),
                "documentary-frame-analysis-v2",
            ]
        )
        return f"{legacy_prefix}_{utils.md5(payload)}"

    def _strip_code_fence(self, response_text: str) -> str:
        cleaned = (response_text or "").strip()
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_batch_response(
        self,
        *,
        batch_index: int,
        raw_response: str,
        frame_paths: list[str],
        time_range: str,
    ) -> FrameBatchResult:
        try:
            payload = json.loads(self._strip_code_fence(raw_response))
        except Exception as exc:
            return self._build_failed_batch_result(
                batch_index=batch_index,
                raw_response=raw_response,
                error_message=str(exc),
                frame_paths=frame_paths,
                time_range=time_range,
            )

        validation_error = self._validate_batch_payload_contract(payload, expected_frame_count=len(frame_paths))
        if validation_error:
            return self._build_failed_batch_result(
                batch_index=batch_index,
                raw_response=raw_response,
                error_message=validation_error,
                frame_paths=frame_paths,
                time_range=time_range,
            )

        raw_observations = payload["frame_observations"]

        frame_observations: list[dict] = []
        for index, frame_path in enumerate(frame_paths):
            entry = raw_observations[index] if index < len(raw_observations) else {}
            if isinstance(entry, dict):
                observation = str(entry.get("observation", "") or "")
                timestamp = str(entry.get("timestamp", "") or "")
            else:
                observation = str(entry or "")
                timestamp = ""
            frame_observations.append(
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "observation": observation,
                }
            )

        summary = payload["overall_activity_summary"]

        return FrameBatchResult(
            batch_index=batch_index,
            status="success",
            time_range=time_range,
            raw_response=raw_response,
            frame_paths=list(frame_paths),
            frame_observations=frame_observations,
            overall_activity_summary=summary,
        )

    def _validate_batch_payload_contract(self, payload: object, *, expected_frame_count: int) -> str:
        if not isinstance(payload, dict):
            return "Batch response JSON payload must be an object"

        if "frame_observations" not in payload or not isinstance(payload["frame_observations"], list):
            return "Batch response must include frame_observations as a list"

        if len(payload["frame_observations"]) < expected_frame_count:
            return (
                "Batch response frame_observations length is shorter than provided frame_paths: "
                f"{len(payload['frame_observations'])} < {expected_frame_count}"
            )

        if "overall_activity_summary" not in payload:
            return "Batch response must include overall_activity_summary"

        if not isinstance(payload["overall_activity_summary"], str):
            return "Batch response overall_activity_summary must be a string"

        return ""

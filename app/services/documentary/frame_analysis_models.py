from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentaryAnalysisConfig:
    video_path: str
    frame_interval_seconds: float
    vision_batch_size: int
    vision_llm_provider: str
    vision_model_name: str
    custom_prompt: str = ""
    max_concurrency: int = 2


@dataclass(slots=True)
class FrameBatchResult:
    batch_index: int
    status: str
    time_range: str
    raw_response: str
    frame_paths: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    summary: str = ""
    fallback_summary: str = ""
    error_message: str = ""

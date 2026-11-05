"""State for one WebSocket session; created per handler call, never shared."""

from dataclasses import dataclass, field

from serving.services.segmenter import KeypointSegmenter


@dataclass
class SessionState:
    segmenter: KeypointSegmenter
    gloss_seq: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.gloss_seq.clear()
        self.segmenter.reset()

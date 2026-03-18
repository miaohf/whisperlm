"""LLM 语义优化与翻译服务"""

import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from ..config import Settings, get_settings
from ..api.models import SegmentResponse


class LLMService:
    """LLM 优化服务，封装语义断句、ASR 纠错、翻译等能力"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # 基础属性与连接
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self.settings.llm.enabled

    def _get_client(self) -> AsyncOpenAI:
        """获取（懒加载）AsyncOpenAI 客户端"""
        if self._client is None:
            cfg = self.settings.llm
            self._client = AsyncOpenAI(
                api_key=cfg.api_key or "dummy",
                base_url=cfg.base_url,
                timeout=float(cfg.timeout),
                max_retries=cfg.max_retries,
            )
        return self._client

    async def check_connection(self) -> bool:
        """检测 LLM 服务是否可达"""
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception as e:
            logger.warning(f"LLM connection check failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # 内部工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_json_response(self, content: str) -> Any:
        """从 LLM 响应中提取 JSON，兼容代码块包裹和前缀文字"""
        # 优先从 ```json ... ``` 代码块中提取
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if code_block:
            content = code_block.group(1)
        else:
            # 找到第一个 [ 或 { 开始的位置，跳过前缀说明文字
            match = re.search(r"[\[\{]", content)
            if match:
                content = content[match.start():]

        return json.loads(content.strip())

    def _build_optimize_prompt(self, segments: list[SegmentResponse], language: str) -> str:
        from ..utils.prompts import OPTIMIZE_PROMPT

        features_cfg = self.settings.llm.features
        feature_lines: list[str] = []
        if features_cfg.error_correction:
            feature_lines.append("修复 ASR 错误（错别字、漏字、同音混淆词等）")
        if features_cfg.expression_optimization:
            feature_lines.append("优化口语表达，使其更加自然流畅")
        if features_cfg.semantic_segmentation:
            feature_lines.append("语义紧密的相邻段落可以合并，时间戳取合并范围的最小/最大值")
        if not feature_lines:
            feature_lines.append("原样返回，不做任何修改")

        input_data = [
            {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
                "speaker": seg.speaker,
            }
            for seg in segments
        ]

        return OPTIMIZE_PROMPT.format(
            language=language or "自动检测",
            features="\n".join(f"{i + 1}. {f}" for i, f in enumerate(feature_lines)),
            segments=json.dumps(input_data, ensure_ascii=False, indent=2),
        )

    def _merge_optimize_result(
        self,
        original: list[SegmentResponse],
        optimized_data: list[dict[str, Any]],
    ) -> list[SegmentResponse]:
        """
        将 LLM 返回的优化数据与原始 segments 合并。
        - 若 id 与原始一致且文本未变化，保留原始 words / confidence
        - 若文本发生变化（含合并），清空 words，保留 confidence
        """
        original_map = {seg.id: seg for seg in original}
        result: list[SegmentResponse] = []

        for i, item in enumerate(optimized_data):
            orig_id = item.get("id")
            orig_seg = original_map.get(orig_id)
            new_text = str(item.get("text", "")).strip()

            if orig_seg and new_text == orig_seg.text:
                words = orig_seg.words
                confidence = orig_seg.confidence
            else:
                words = []
                confidence = orig_seg.confidence if orig_seg else 0.0

            result.append(
                SegmentResponse(
                    id=i,
                    start=float(item.get("start", orig_seg.start if orig_seg else 0.0)),
                    end=float(item.get("end", orig_seg.end if orig_seg else 0.0)),
                    text=new_text,
                    speaker=item.get("speaker", orig_seg.speaker if orig_seg else None),
                    words=words,
                    confidence=confidence,
                )
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────────────────

    async def optimize(
        self,
        segments: list[SegmentResponse],
        language: str | None = None,
    ) -> list[SegmentResponse]:
        """
        使用 LLM 对转录段落进行语义优化（错误修复 / 表达优化 / 语义断句）。
        若 LLM 不可用或处理失败，原样返回原始段落，不抛出异常。
        """
        if not self.is_enabled or not segments:
            return segments

        features = self.settings.llm.features
        if not any([
            features.semantic_segmentation,
            features.error_correction,
            features.expression_optimization,
        ]):
            return segments

        prompt = self._build_optimize_prompt(segments, language or "auto")

        try:
            client = self._get_client()
            logger.info(
                f"LLM optimize: model={self.settings.llm.model}, "
                f"segments={len(segments)}, language={language or 'auto'}"
            )
            response = await client.chat.completions.create(
                model=self.settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM optimize raw response ({len(content)} chars): {content[:300]}...")

            optimized_data: list[dict] = self._parse_json_response(content)
            if not isinstance(optimized_data, list):
                raise ValueError(f"Expected list, got {type(optimized_data)}")

            result = self._merge_optimize_result(segments, optimized_data)
            logger.info(
                f"LLM optimize completed: {len(segments)} -> {len(result)} segments"
            )
            return result

        except Exception as e:
            logger.warning(
                f"LLM optimization failed, returning original segments. Error: {e}"
            )
            return segments

    async def translate(
        self,
        segments: list[SegmentResponse],
        target_language: str,
        style: str = "natural",
    ) -> list[SegmentResponse]:
        """
        使用 LLM 翻译段落文本，结果写入 translated_text 字段。
        若 LLM 不可用或处理失败，原样返回原始段落（translated_text 为 None）。
        """
        if not segments:
            return segments

        from ..utils.prompts import TRANSLATION_PROMPT, STYLE_DESCRIPTIONS

        style_desc = STYLE_DESCRIPTIONS.get(style, style)
        input_data = [{"id": seg.id, "text": seg.text} for seg in segments]

        prompt = TRANSLATION_PROMPT.format(
            target_language=target_language,
            style_desc=style_desc,
            segments=json.dumps(input_data, ensure_ascii=False, indent=2),
        )

        try:
            client = self._get_client()
            logger.info(
                f"LLM translate: model={self.settings.llm.model}, "
                f"segments={len(segments)}, target={target_language}, style={style}"
            )
            response = await client.chat.completions.create(
                model=self.settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8192,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM translate raw response ({len(content)} chars): {content[:300]}...")

            translations: list[dict] = self._parse_json_response(content)
            if not isinstance(translations, list):
                raise ValueError(f"Expected list, got {type(translations)}")

            translation_map = {
                item["id"]: item.get("translated_text", "")
                for item in translations
                if "id" in item
            }

            result: list[SegmentResponse] = []
            for seg in segments:
                updated = seg.model_copy(
                    update={"translated_text": translation_map.get(seg.id)}
                )
                result.append(updated)

            logger.info(
                f"LLM translate completed: {len(segments)} segments -> {target_language}"
            )
            return result

        except Exception as e:
            logger.warning(
                f"LLM translation failed, returning original segments. Error: {e}"
            )
            return segments

"""Verify interview segmentation and scoring across several batches."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from agent.llm_client import LLMJSONError
from agent.pipeline import _score_interview_batch, analyze_interview


class InterviewAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_interview_scores_every_unique_question_in_order(self):
        transcript = "甲方：第一段\n乙方：第二段"
        questions = [{"question": f"问题 {index}"} for index in range(1, 8)]
        llm = AsyncMock()
        llm.model = "test-model"
        llm.chat_json.side_effect = [
            {"questions": questions[:4]},
            {"questions": questions[3:]},
            {"overall_summary": "前六题", "question_analyses": [
                {"id": str(index), "score": 80} for index in range(6, 0, -1)
            ]},
            {"overall_summary": "最后一题", "question_analyses": [{"id": 1, "score": 87}]},
        ]
        with (
            patch("agent.pipeline.LLMClient", return_value=llm),
            patch("agent.pipeline.get_context_manager") as context,
            patch("agent.pipeline.TextProcessor.chunk_by_sentences", return_value=["第一段", "第二段"]),
        ):
            context.return_value.count_tokens.return_value = 9000
            result = await analyze_interview(transcript)

        ids = [item["id"] for item in result["a_data"]["question_analyses"]]
        self.assertEqual(ids, list(range(1, 8)))
        self.assertEqual(result["a_data"]["overall_score"], 81)
        self.assertEqual(len(result["q_data"]["questions"]), 7)
        self.assertEqual(llm.chat_json.await_count, 4)

    async def test_missing_batch_positions_retries_smaller_batches(self):
        llm = AsyncMock()
        llm.chat_json.side_effect = [
            {"question_analyses": [{"id": 1, "score": 20}]},
            {"overall_summary": "第一题", "question_analyses": [{"score": 80}]},
            {"overall_summary": "第二题", "question_analyses": [{"id": 1, "score": 90}]},
        ]
        analyses, summaries = await _score_interview_batch(
            llm, "模拟转写", ({"id": 13, "question": "甲"}, {"id": 14, "question": "乙"})
        )
        self.assertEqual([(item["id"], item["score"]) for item in analyses], [(13, 80), (14, 90)])
        self.assertEqual(summaries, ["第一题", "第二题"])
        self.assertEqual(llm.chat_json.await_count, 3)

    async def test_extraction_failure_does_not_silently_omit_a_segment(self):
        llm = AsyncMock()
        llm.chat_json.side_effect = [{"questions": [{"question": "问题一"}]}, LLMJSONError("length")]
        with (
            patch("agent.pipeline.LLMClient", return_value=llm),
            patch("agent.pipeline.get_context_manager") as context,
            patch("agent.pipeline.TextProcessor.chunk_by_sentences", return_value=["第一段", "第二段"]),
        ):
            context.return_value.count_tokens.return_value = 9000
            with self.assertRaises(LLMJSONError):
                await analyze_interview("长录音")

        self.assertEqual(llm.chat_json.await_count, 2)

    async def test_failed_fallback_is_not_run_twice(self):
        from agent.graph import agent_loop

        llm = AsyncMock()
        llm.chat_with_tools.return_value = {"content": "", "tool_calls": None}
        state = {
            "task_id": "test-interview",
            "audio_stem": "test-interview",
            "source_type": "local_audio",
            "transcript_text": "模拟面试",
            "asr_duration": 1,
        }
        with (
            patch("agent.llm_client.LLMClient", return_value=llm),
            patch("agent.graph.get_context_manager") as context,
            patch("agent.graph._fallback_classify", return_value="interview") as classify,
            patch("agent.graph._fallback_analyze", side_effect=LLMJSONError("length")) as analyze,
        ):
            context.return_value.count_tokens.return_value = 4
            with self.assertRaises(LLMJSONError):
                await agent_loop(state)

        classify.assert_awaited_once()
        analyze.assert_awaited_once()

    async def test_interview_tool_reads_transcript_from_context(self):
        from agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, ToolContext

        schema = next(tool["function"] for tool in TOOL_SCHEMAS
                      if tool["function"]["name"] == "analyze_interview")
        self.assertEqual(schema["parameters"]["properties"], {})
        context = ToolContext("完整转写", "sample", llm_model="fake")
        with (
            TemporaryDirectory() as output,
            patch("agent.cache.load_interview_cache", return_value=None),
            patch("agent.pipeline.settings.output_dir", Path(output)),
            patch("agent.pipeline.analyze_interview", return_value={
                "q_data": {"questions": []},
                "a_data": {"overall_summary": "测试"},
                "llm_model_used": "fake",
            }) as analyze,
        ):
            result = await TOOL_REGISTRY["analyze_interview"]({"transcript": "模型转述"}, context)
            self.assertTrue(Path(result["result_path"]).is_file())

        analyze.assert_awaited_once_with("完整转写", llm_model="fake")

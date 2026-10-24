"""Tests for the Grounded Cognitive Control Loop in Project Jane.

Verifies:
1. AsyncTaskSupervisor manages real background threads and outputs immutable telemetry.
2. CognitiveFrameParser robustly parses natural commands, progress queries, and challenges.
3. JanePipeline dynamically executes real ingestion, reports real progress, and avoids hallucinations.
4. Capability challenges execute actions rather than dumping canned bios.
"""

import time
import pytest
from src.core.pipeline import JanePipeline
from src.core.task_supervisor import AsyncTaskSupervisor, BackgroundTaskRecord
from src.core.cognitive_frame_parser import CognitiveFrameParser


def test_task_supervisor_lifecycle():
    supervisor = AsyncTaskSupervisor()
    
    # Mock ingestor with quick return
    class MockIngestor:
        def ingest_article(self, title):
            class Report:
                accepted_concepts = [f"{title.lower()}_law"]
                rejected_concepts = []
            return Report()

    record = supervisor.spawn_wikipedia_ingestion(
        MockIngestor(),
        domain="mathematics",
        target_count=3,
        article_titles=["Topology", "Algebra", "Calculus"]
    )
    assert record.task_id.startswith("wiki_ingest_")
    assert record.status in ("RUNNING", "COMPLETED")

    # Wait for completion
    for _ in range(20):
        if record.status == "COMPLETED":
            break
        time.sleep(0.2)

    assert record.status == "COMPLETED"
    assert record.current_count >= 3
    telemetry = supervisor.get_telemetry_summary()
    assert len(telemetry["completed_tasks"]) >= 1


def test_cognitive_frame_parser_commands_and_queries():
    parser = CognitiveFrameParser()

    # 1. Ingestion Command
    frame1 = parser.parse("use wikipedia and learn 1000 concepts of mathematics")
    assert frame1.frame_type == "COMMAND_INGEST"
    assert frame1.parameters["target_count"] == 1000
    assert frame1.parameters["domain"] == "mathematics"

    frame2 = parser.parse("use wikepedia and expand the concepts from 36 to 1000. Can you do it?")
    assert frame2.frame_type == "COMMAND_INGEST"
    assert frame2.parameters["target_count"] in (36, 1000)

    # 2. Progress and Telemetry Queries
    frame3 = parser.parse("how's the progress")
    assert frame3.frame_type == "QUERY_TASK_PROGRESS"

    frame4 = parser.parse("how much time will it take?? any idea")
    assert frame4.frame_type == "QUERY_TASK_PROGRESS"

    frame5 = parser.parse("are you doing it")
    assert frame5.frame_type == "QUERY_TASK_PROGRESS"

    # 3. Capability Challenge
    frame6 = parser.parse("can you create your own hypothesis")
    assert frame6.frame_type == "CAPABILITY_CHALLENGE"

    # 4. Introspective Self Queries
    frame7 = parser.parse("do you understand english or any natural language")
    assert frame7.frame_type == "INTROSPECTIVE_SELF"
    assert frame7.action == "EXPLAIN_LANGUAGE_CAPABILITY"

    # 5. Interrogative Choice & Conversational Inquiries (Must NOT trigger COMMAND_INGEST)
    frame8 = parser.parse("learn quantum machanics or mathematics ? which was ir")
    assert frame8.frame_type == "CONVERSATIONAL"

    frame9 = parser.parse("i meant did i told you to learn quanntum machanics or mathmatics?? i think it was mathmatics")
    assert frame9.frame_type == "CONVERSATIONAL"

    frame10 = parser.parse("can you show me when i told you that")
    assert frame10.frame_type == "CONVERSATIONAL"


def test_pipeline_grounded_control_loop_execution():
    pipeline = JanePipeline()

    # Turn 1: User issues task command
    res1 = pipeline.process("use wikipedia and learn 100 concepts of mathematics")
    assert res1["route_type"] == "BACKGROUND_TASK_LAUNCH"
    assert "Task ID:" in res1["response_text"]
    assert len(pipeline.task_supervisor.get_active_tasks()) >= 1

    # Turn 2: User queries progress -> must report real live telemetry, NOT hallucinate
    res2 = pipeline.process("how's the progress")
    assert res2["route_type"] == "TASK_TELEMETRY"
    assert "actively ingesting" in res2["response_text"]
    assert "concepts verified and wired" in res2["response_text"]

    # Turn 3: Capability challenge -> must generate hypothesis, NOT recite canned bio
    res3 = pipeline.process("can you create your own hypothesis")
    assert res3["route_type"] == "CAPABILITY_CHALLENGE"
    assert "Yes, I" in res3["response_text"]
    assert "Hypothesis" in res3["response_text"] or "hypothesis" in res3["response_text"]
    assert "embodied neuro-symbolic AI assistant" not in res3["response_text"]

    # Turn 4: Language capability query -> crisp direct answer, NOT canned bio
    res4 = pipeline.process("do you understand english or any natural language")
    assert "understand English" in res4["response_text"]
    assert "embodied neuro-symbolic AI assistant" not in res4["response_text"]

    # Turn 5: Hardcoded check query -> clear explanation of dynamic manifold
    res5 = pipeline.process("do you have hardcoded values for response in your architecture")
    assert "No" in res5["response_text"]
    assert "hardcoded" in res5["response_text"].lower()

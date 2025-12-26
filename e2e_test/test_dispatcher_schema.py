"""Tests for dispatcher structured output parsing.

Tests issue hi_moe-4dy: Dispatcher structured output via prompt enforcement.
"""
import pytest

from .dispatcher_schema import (
    Step,
    DispatcherPlan,
    extract_json_from_response,
    parse_dispatcher_response,
    get_dispatcher_plan,
    VALID_SPECIALISTS,
)


class TestStep:
    """Tests for Step dataclass."""

    def test_valid_step(self):
        step = Step(description="Write code", specialist="python")
        assert step.description == "Write code"
        assert step.specialist == "python"

    def test_empty_description_fails(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Step(description="", specialist="python")

    def test_whitespace_description_fails(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Step(description="   ", specialist="python")

    def test_invalid_specialist_fails(self):
        with pytest.raises(ValueError, match="Invalid specialist"):
            Step(description="Test", specialist="invalid")

    def test_all_valid_specialists(self):
        for specialist in VALID_SPECIALISTS:
            step = Step(description="Test", specialist=specialist)
            assert step.specialist == specialist


class TestDispatcherPlan:
    """Tests for DispatcherPlan dataclass."""

    def test_valid_plan(self):
        plan = DispatcherPlan(
            steps=[Step(description="Step 1", specialist="python")]
        )
        assert len(plan.steps) == 1

    def test_empty_steps_fails(self):
        with pytest.raises(ValueError, match="at least one step"):
            DispatcherPlan(steps=[])

    def test_from_dict_valid(self):
        data = {
            "steps": [
                {"description": "Analyze problem", "specialist": "math"},
                {"description": "Implement solution", "specialist": "python"},
            ]
        }
        plan = DispatcherPlan.from_dict(data)
        assert len(plan.steps) == 2
        assert plan.steps[0].specialist == "math"
        assert plan.steps[1].specialist == "python"

    def test_from_dict_missing_steps(self):
        with pytest.raises(ValueError, match="Missing 'steps' key"):
            DispatcherPlan.from_dict({})

    def test_from_dict_steps_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            DispatcherPlan.from_dict({"steps": "not a list"})

    def test_from_dict_step_not_object(self):
        with pytest.raises(ValueError, match="must be an object"):
            DispatcherPlan.from_dict({"steps": ["not an object"]})

    def test_from_dict_missing_description(self):
        with pytest.raises(ValueError, match="missing 'description'"):
            DispatcherPlan.from_dict({"steps": [{"specialist": "python"}]})

    def test_from_dict_missing_specialist(self):
        with pytest.raises(ValueError, match="missing 'specialist'"):
            DispatcherPlan.from_dict({"steps": [{"description": "Test"}]})


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_clean_json(self):
        response = '{"steps": [{"description": "Test", "specialist": "python"}]}'
        result = extract_json_from_response(response)
        assert result == {"steps": [{"description": "Test", "specialist": "python"}]}

    def test_json_with_whitespace(self):
        response = '  \n{"steps": [{"description": "Test", "specialist": "python"}]}\n  '
        result = extract_json_from_response(response)
        assert "steps" in result

    def test_json_in_markdown_block(self):
        response = '''Here's the plan:
```json
{"steps": [{"description": "Test", "specialist": "python"}]}
```'''
        result = extract_json_from_response(response)
        assert "steps" in result

    def test_json_in_generic_code_block(self):
        response = '''```
{"steps": [{"description": "Test", "specialist": "python"}]}
```'''
        result = extract_json_from_response(response)
        assert "steps" in result

    def test_json_embedded_in_text(self):
        response = 'Some preamble {"steps": [{"description": "Test", "specialist": "python"}]} some after'
        result = extract_json_from_response(response)
        assert "steps" in result

    def test_invalid_json_fails(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            extract_json_from_response("This is not JSON at all")

    def test_malformed_json_fails(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            extract_json_from_response('{"steps": [incomplete')


class TestParseDispatcherResponse:
    """Tests for full response parsing."""

    def test_valid_response(self):
        response = '{"steps": [{"description": "Implement", "specialist": "python"}]}'
        plan = parse_dispatcher_response(response)
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Implement"

    def test_multi_step_response(self):
        response = '''{"steps": [
            {"description": "Analyze", "specialist": "math"},
            {"description": "Implement", "specialist": "python"},
            {"description": "Review", "specialist": "general"}
        ]}'''
        plan = parse_dispatcher_response(response)
        assert len(plan.steps) == 3

    def test_invalid_structure_fails(self):
        with pytest.raises(ValueError):
            parse_dispatcher_response('{"not_steps": []}')


class TestGetDispatcherPlan:
    """Tests for the full plan generation with retry."""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful plan generation with mock client."""

        class MockClient:
            async def generate(self, messages, **kwargs):
                return '{"steps": [{"description": "Test step", "specialist": "python"}]}'

        plan = await get_dispatcher_plan(MockClient(), "Solve a problem")
        assert len(plan.steps) == 1
        assert plan.steps[0].specialist == "python"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that retry works when first attempt fails."""

        class RetryClient:
            def __init__(self):
                self.attempts = 0

            async def generate(self, messages, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    return "Invalid JSON response"
                return '{"steps": [{"description": "Success on retry", "specialist": "python"}]}'

        client = RetryClient()
        plan = await get_dispatcher_plan(client, "Test task", max_retries=1)
        assert client.attempts == 2
        assert plan.steps[0].description == "Success on retry"

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self):
        """Test that it fails after exhausting retries."""

        class AlwaysFailClient:
            async def generate(self, messages, **kwargs):
                return "Never valid JSON"

        with pytest.raises(ValueError, match="Failed to parse"):
            await get_dispatcher_plan(
                AlwaysFailClient(), "Test", max_retries=1
            )

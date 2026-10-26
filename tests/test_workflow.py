from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

import pipeline
from models import CompetitorReport, DiscoveryResult, ResearchBundle, ResearchState
from search_tools import SearchTools, normalize_results
from errors import describe_error

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_model_error_is_actionable_without_exposing_response():
    import httpx
    from openai import NotFoundError
    response = httpx.Response(404, request=httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1"))
    error = NotFoundError("private-request-data", response=response,
                         body={"error": {"code": "model_not_found", "message": "private-request-data"}})
    message = describe_error(error)
    assert "Nebius model unavailable" in message
    assert "NEBIUS_MODEL" in message
    assert "private-request-data" not in message


def test_provider_key_errors_do_not_expose_keys():
    from tavily.errors import InvalidAPIKeyError
    message = describe_error(InvalidAPIKeyError("private-api-key"))
    assert "Tavily rejected the API key" in message
    assert "private-api-key" not in message


@pytest.mark.parametrize("status, explanation", [
    (413, "size or context limit"),
    (424, "could not serve"),
    (498, "could not serve"),
    (503, "server error"),
    (422, "rejected the request"),
    (418, "Report this HTTP status"),
])
def test_api_status_errors_show_status_without_private_payload(status, explanation):
    import httpx
    from openai import APIStatusError
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1"))
    error = APIStatusError("private-api-key", response=response,
                           body={"error": {"message": "private-request-data"}})
    message = describe_error(error)
    assert f"Nebius request failed (HTTP {status}, APIStatusError)" in message
    assert explanation in message
    assert "private-api-key" not in message
    assert "private-request-data" not in message


def discovery():
    return DiscoveryResult(company_name="Target", primary_market="Design software",
                           competitors=[{"name": name, "reason": "Product overlap"}
                                        for name in ["Alpha", "Beta", "Gamma"]])


def report(name="Alpha", **kwargs):
    fields = dict(competitor_name=name, pricing_model="Subscription; exact price not found",
                  core_features=["Editing", "Sharing", "Templates"],
                  market_positioning="Design tools for small teams", recent_news=[],
                  source_urls=["https://example.com/pricing"])
    return CompetitorReport(**(fields | kwargs))


def bundle(name="Alpha"):
    return ResearchBundle(competitor_name=name, web_results=[{
        "title": "Pricing", "url": "https://example.com/pricing",
        "content": "Subscriptions for editing, sharing, templates.",
    }], news_results=[])


def research_state(name="Alpha") -> ResearchState:
    return {"company": "Target", "competitor_queue": [], "current_target": name,
            "raw_data": bundle(name).model_dump_json(), "final_reports": [], "messages": []}


def test_normalization_preserves_content_and_null_date():
    result = normalize_results({"results": [{"title": "Page", "url": "https://example.com",
                                              "content": "raw content", "score": 0.9}]})
    assert result == [{"title": "Page", "url": "https://example.com", "content": "raw content",
                       "published_date": None}]
    assert normalize_results({"results": [{"url": "", "content": "bad"}]}) == []


def test_tools_search_both_topics_and_keep_raw_records():
    client = Mock()
    client.search.return_value = {"results": [bundle().web_results[0].model_dump()]}
    searches = SearchTools(client, news_days=7)
    searches.web.invoke({"query": "Alpha pricing"})
    searches.news.invoke({"query": "Alpha launches"})
    searches.web.invoke({"query": "Alpha pricing"})
    assert client.search.call_count == 2
    assert client.search.call_args_list[1].kwargs["topic"] == "news"
    assert client.search.call_args_list[1].kwargs["days"] == 7
    assert searches.records["general"][0]["content"] == bundle().web_results[0].content


def test_schemas_reject_wrong_competitor_count_duplicates_and_features():
    payload = discovery().model_dump()
    with pytest.raises(ValidationError):
        DiscoveryResult(**(payload | {"competitors": payload["competitors"][:2]}))
    with pytest.raises(ValidationError):
        DiscoveryResult(**(payload | {"competitors": [payload["competitors"][0]] * 3}))
    with pytest.raises(ValidationError):
        report(core_features=["Only one"])


def test_pipeline_keeps_successes_when_one_competitor_fails():
    runner = object.__new__(pipeline.MarketResearchPipeline)
    runner.discover = Mock(return_value=discovery())
    runner.research = Mock(side_effect=[bundle("Alpha"), RuntimeError("private provider error"), bundle("Gamma")])
    runner.analyze = Mock(side_effect=[report("Alpha"), report("Gamma")])
    result = runner.run(" Target ")
    assert [item.competitor_name for item in result.reports] == ["Alpha", "Gamma"]
    assert list(result.errors) == ["Beta"]
    assert "private provider error" not in result.model_dump_json()
    research_states = [call.args[0] for call in runner.research.call_args_list]
    assert [state["current_target"] for state in research_states] == ["Alpha", "Beta", "Gamma"]
    assert all(state["raw_data"] == "" for state in research_states)
    assert [state["competitor_queue"] for state in research_states] == [["Beta", "Gamma"], ["Gamma"], []]
    assert [call.args[0]["current_target"] for call in runner.analyze.call_args_list] == ["Alpha", "Gamma"]


def test_all_agents_share_state_and_reducer_accumulates_once(monkeypatch):
    runner = object.__new__(pipeline.MarketResearchPipeline)
    runner.model, runner.news_days, runner.client = Mock(), 30, Mock()
    runner.client.search.return_value = {"results": [bundle().web_results[0].model_dump()]}
    agents = []

    def make_agent(**kwargs):
        assert kwargs["state_schema"] is ResearchState
        agent = Mock()
        index = len(agents) % 4
        if index == 0:
            agent.invoke.return_value = {"structured_response": discovery()}
        else:
            agent.invoke.return_value = {"structured_response": report(["Alpha", "Beta", "Gamma"][index - 1])}
        agents.append(agent)
        return agent

    monkeypatch.setattr(pipeline, "create_agent", make_agent)
    # Two separate runs must not reuse accumulated reports or queue contents.
    for _ in range(2):
        result = runner.run("Target")
        assert [item.competitor_name for item in result.reports] == ["Alpha", "Beta", "Gamma"]
        assert not result.errors
        run_agents = agents[-4:]
        for index, name in zip([1, 2, 3], ["Alpha", "Beta", "Gamma"]):
            state = run_agents[index].invoke.call_args.args[0]
            assert state["company"] == "Target"
            assert state["current_target"] == name
            assert ResearchBundle.model_validate_json(state["raw_data"]).competitor_name == name
            assert len(state["final_reports"]) == index - 1


def test_analysis_rejects_invented_sources(monkeypatch):
    runner = object.__new__(pipeline.MarketResearchPipeline)
    runner.model, runner.news_days = Mock(), 30
    agent = Mock()
    agent.invoke.return_value = {"structured_response": report(source_urls=["https://invented.com"])}
    monkeypatch.setattr(pipeline, "create_agent", Mock(return_value=agent))
    with pytest.raises(ValueError, match="outside the captured evidence"):
        runner.analyze(research_state())


def test_research_fetches_both_topics_without_agent_or_model(monkeypatch):
    runner = object.__new__(pipeline.MarketResearchPipeline)
    runner.news_days, runner.client = 30, Mock()
    runner.client.search.return_value = {"results": [bundle().web_results[0].model_dump()]}
    factory = Mock(side_effect=AssertionError("Research must not create an agent"))
    monkeypatch.setattr(pipeline, "create_agent", factory)
    result = runner.research(research_state(), "Design software")
    factory.assert_not_called()
    assert [call.kwargs["topic"] for call in runner.client.search.call_args_list] == ["general", "news"]
    assert len(result.web_results) == len(result.news_results) == 1
    assert result.web_results[0].content == bundle().web_results[0].content


def test_news_recency():
    now = datetime.now(timezone.utc)
    assert pipeline.is_recent_date((now - timedelta(days=1)).isoformat(), 30)
    assert not pipeline.is_recent_date((now - timedelta(days=40)).isoformat(), 30)
    assert not pipeline.is_recent_date((now + timedelta(days=1)).isoformat(), 30)
    assert not pipeline.is_recent_date("unknown", 30)


def test_streamlit_loads_and_validates_empty_input(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    app.button[0].click().run()
    assert app.warning[0].value == "Enter a company name."


def test_streamlit_renders_report_cards_and_empty_news():
    from models import MarketReport
    app = AppTest.from_file(APP_PATH)
    app.session_state["report"] = MarketReport(discovery=discovery(),
        reports=[report()], research=[bundle()], errors={})
    app.run()
    assert not app.exception
    assert any(item.value == "Alpha" for item in app.subheader)
    content = " ".join(item.value for item in app.markdown)
    assert "Market positioning" in content
    assert "Recent news" in content
    assert "No dated news found" in content

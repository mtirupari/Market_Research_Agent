from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from pipeline import MarketResearchPipeline
from errors import describe_error

config = dotenv_values(Path(__file__).resolve().with_name(".env"))
nebius_key = (config.get("NEBIUS_API_KEY") or "").strip()
tavily_key = (config.get("TAVILY_API_KEY") or "").strip()
base_url = (config.get("NEBIUS_BASE_URL") or "https://api.tokenfactory.nebius.com/v1/").strip()
st.set_page_config(page_title="Market Research Agent", page_icon="🔎", layout="wide")
st.title("Market Research Agent")
st.write("Discover three competitors and compare their pricing, features, positioning, and latest news.")


with st.sidebar:
    st.header("Research settings")
    model = st.text_input("Nebius model", value=config.get("NEBIUS_MODEL") or "Qwen/Qwen3.5-397B-A17B")
    news_days = st.selectbox("Recent news window (days)", [7, 30, 90], index=1)
    st.caption("Each run uses the configured Nebius and Tavily API quotas.")

with st.form("research"):
    company = st.text_input("Company name", placeholder="e.g. Canva, Nike, or Salesforce", max_chars=200)
    submitted = st.form_submit_button("Research competitors", type="primary")

if submitted:
    if not company.strip():
        st.warning("Enter a company name.")
    elif not nebius_key or not tavily_key:
        st.warning("Set NEBIUS_API_KEY and TAVILY_API_KEY in the project's .env file before running research.")
    elif not model.strip():
        st.warning("Provide a Nebius model in the sidebar.")
    else:
        st.session_state.pop("report", None)
        with st.status("Starting market research…", expanded=True) as status:
            try:
                pipeline = MarketResearchPipeline(nebius_key.strip(), tavily_key.strip(), model.strip(), news_days,
                                                  base_url=base_url)
                result = pipeline.run(company, progress=status.write)
                st.session_state.report = result
                status.update(label="Research complete" if not result.errors else "Research finished with incomplete results",
                              state="complete" if not result.errors else "error", expanded=False)
            except Exception as error:
                status.update(label="Research could not complete", state="error")
                st.error(describe_error(error))

if "report" in st.session_state:
    result = st.session_state.report
    st.subheader(f"{result.discovery.company_name} · {result.discovery.primary_market}")
    st.caption("Competitors are selected by product and market fit, not a verified market-share ranking.")
    for competitor in result.discovery.competitors:
        st.write(f"**{competitor.name}** — {competitor.reason}")
    for name, error in result.errors.items():
        st.warning(f"{name}: {error}")
    for report in result.reports:
        with st.container(border=True):
            st.subheader(report.competitor_name)
            left, right = st.columns(2)
            with left:
                st.markdown("**Pricing model**")
                st.write(report.pricing_model)
                st.markdown("**Market positioning**")
                st.write(report.market_positioning)
            with right:
                st.markdown("**Core features**")
                for feature in report.core_features:
                    st.write(f"• {feature}")
            st.markdown("**Recent news**")
            if not report.recent_news:
                st.write("No dated news found in the selected window.")
            for news in report.recent_news:
                st.write(f"**{news.title}** · {news.published_date}")
                st.write(news.summary)
                st.write(news.url)
            with st.expander("Sources"):
                for url in report.source_urls:
                    st.write(url)
    with st.expander("Raw research data"):
        st.json([bundle.model_dump() for bundle in result.research])
    st.download_button("Download JSON report", result.model_dump_json(indent=2),
                       file_name="competitor_report.json", mime="application/json")

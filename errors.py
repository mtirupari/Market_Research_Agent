"""Actionable provider errors without exposing request bodies or credentials."""


class ResearchDataError(ValueError):
    """An application-controlled validation message, safe to display."""


def describe_error(error: Exception) -> str:
    if isinstance(error, ResearchDataError):
        return str(error)
    module = type(error).__module__
    name = type(error).__name__
    provider = "Nebius" if module.startswith(("openai", "langchain_nebius")) else "Tavily" if module.startswith("tavily") else "Provider"
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    code = detail.get("code") if isinstance(detail, dict) else None
    if code in ("model_not_found", "model_decommissioned") or (provider == "Nebius" and status == 404):
        return "Nebius model unavailable. Set NEBIUS_MODEL in .env to a model your account can access, then reload the app."
    if status == 401 or name in ("InvalidAPIKeyError", "MissingAPIKeyError", "AuthenticationError"):
        return f"{provider} rejected the API key. Check the corresponding key in .env."
    if status == 429 or name in ("UsageLimitExceededError", "RateLimitError"):
        return f"{provider} rate or usage limit reached. Wait before retrying, or check your provider quota."
    if status == 403 or name in ("ForbiddenError", "PermissionDeniedError"):
        return f"{provider} denied access. Check account permissions and model access."
    if code == "tool_use_failed":
        return "Nebius could not produce a valid tool call. Retry or choose another model supporting tool calls."
    if isinstance(status, int):
        prefix = f"{provider} request failed (HTTP {status}, {name}). "
        if status == 402:
            return prefix + "Check provider billing and available credits."
        if status == 413 or code == "context_length_exceeded":
            return prefix + "The request exceeds the provider's size or context limit. Reduce the search content sent to the model."
        if status in (408, 504):
            return prefix + "The provider timed out. Retry shortly."
        if status in (424, 498):
            return prefix + "The provider could not serve this request. Retry shortly or select another available model."
        if status >= 500:
            return prefix + "The provider encountered a server error. Retry shortly; if it persists, check provider service status."
        if status in (400, 422):
            return prefix + "The provider rejected the request. Check model support for tools and structured output."
        return prefix + "Report this HTTP status for diagnosis if retrying does not help."
    if "Timeout" in name:
        return f"{provider} request timed out. Retry shortly."
    if "Connection" in name:
        return f"Could not connect to {provider.lower()}. Check network access and retry."
    if name == "GraphRecursionError":
        return "The agent reached its step limit before producing a valid report. Retry or choose another model."
    # Do not include str(error): it can contain credentials or full API requests.
    return f"Research failed ({name}). Retry; if it persists, report this error type for diagnosis."

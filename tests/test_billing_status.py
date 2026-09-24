"""Provider billing/auth refusals surface as 402 + x-saint-error: billing, not a generic 502."""
from saint.dispatch import is_billing_failure
from saint import server as S


def test_is_billing_failure_markers_and_status():
    assert is_billing_failure("AnthropicException - Your credit balance is too low to access the Anthropic API.", 400)
    assert is_billing_failure("insufficient credits", None)
    assert is_billing_failure("whatever", 402) and is_billing_failure("nope", 401)
    assert not is_billing_failure("Connection error.", None)
    assert not is_billing_failure("max_tokens must be > 0", 400)


def test_billing_response_shapes():
    assert S._is_billing("BillingError: Your credit balance is too low") and not S._is_billing("APIConnectionError")
    r = S._billing_response("BillingError: Your credit balance is too low")
    assert r.status_code == 402 and r.headers["x-saint-error"] == "billing"
    body = r.body.decode()
    assert "billing_error" in body and "credit balance" in body
    a = S._billing_response("BillingError: x", anthropic=True)
    assert a.status_code == 402 and "billing_error" in a.body.decode()

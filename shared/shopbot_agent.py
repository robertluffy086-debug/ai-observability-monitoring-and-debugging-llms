

import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

load_dotenv()

# ---------------------------------------------------------------------------
# Try to reuse the real Ch1 system prompt. Falls back to a local copy if the
# shared config file/import path doesn't match your repo — update as needed.
# ---------------------------------------------------------------------------
try:
    from shared.shopbot_config import SHOPBOT_SYSTEM_PROMPT
except ImportError:
    SHOPBOT_SYSTEM_PROMPT = (
        "You are ShopBot, a customer support assistant for an online retailer. "
        "Answer customer questions using only the tools available to you. "
        "Never guess or make up information about orders, policies, or refunds. "
        "If you don't have the information, say so and offer to escalate."
    )

client = OpenAI()


tracer_provider = register(project_name="shopbot-agent-0721")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)



TODAY = datetime(2026, 7, 18)  # fixed "today" so demo results are reproducible

ORDERS_DB = {
    "ORD-1001": {
        "status": "delivered",
        "delivered_date": (TODAY - timedelta(days=5)).strftime("%Y-%m-%d"),
        "amount": 89.99,
    },
    "ORD-1002": {
        "status": "shipped",
        "delivered_date": None,
        "amount": 45.00,
    },
    "ORD-1003": {
        "status": "delivered",
        "delivered_date": (TODAY - timedelta(days=47)).strftime("%Y-%m-%d"),  # past 30-day window
        "amount": 210.50,
    },
    "ORD-1004": {
        "status": "delivered",
        "delivered_date": (TODAY - timedelta(days=12)).strftime("%Y-%m-%d"),
        "amount": 64.25,
    },
    "ORD-1005": {
        "status": "delivered",
        "delivered_date": (TODAY - timedelta(days=2)).strftime("%Y-%m-%d"),
        "amount": 132.40,
    },
}

RETURN_WINDOW_DAYS = 30

POLICY_DOCS = {
    "return": (
        f"Items may be returned within {RETURN_WINDOW_DAYS} days of delivery "
        "for a full refund, provided the item is unused and in original packaging."
    ),
    "refund": (
        "Approved refunds are processed within 5-7 business days and returned "
        "to the original payment method."
    ),
    "shipping": (
        "Orders over $75 qualify for free standard shipping. Orders under $75 "
        "incur a flat $5.99 shipping fee."
    ),
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def search_policy_docs(query: str) -> dict:
    query_lower = query.lower()
    for keyword, text in POLICY_DOCS.items():
        if keyword in query_lower:
            return {"result": text}
    return {"result": "No matching policy document found for this query."}


def check_order_status(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": f"No order found with ID {order_id}"}
    return order


def check_return_eligibility(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": f"No order found with ID {order_id}"}
    if order["status"] != "delivered":
        return {"eligible": False, "reason": "Order has not been delivered yet."}

    delivered = datetime.strptime(order["delivered_date"], "%Y-%m-%d")
    days_since = (TODAY - delivered).days
    eligible = days_since <= RETURN_WINDOW_DAYS
    return {
        "eligible": eligible,
        "days_since_delivery": days_since,
        "policy_window_days": RETURN_WINDOW_DAYS,
    }


def process_refund(order_id: str, amount: float) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": f"No order found with ID {order_id}"}
    return {"status": "refund_initiated", "order_id": order_id, "amount": amount}


TOOL_IMPL = {
    "search_policy_docs": search_policy_docs,
    "check_order_status": check_order_status,
    "check_return_eligibility": check_return_eligibility,
    "process_refund": process_refund,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_docs",
            "description": "Search ShopBot's policy documents for information about returns, refunds, or shipping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'return policy'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Look up the current status of an order by its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "e.g. 'ORD-1001'"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": "Check whether an order is still within the return window and eligible for return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "e.g. 'ORD-1001'"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Initiate a refund for an order. Only call this after confirming return eligibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Refund amount in USD"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]


def run_shopbot_agent(user_message: str, max_turns: int = 5) -> str:
    with tracer.start_as_current_span("shopbot_agent_turn") as root_span:
        root_span.set_attribute("openinference.span.kind", "AGENT")
        root_span.set_attribute("input.value", user_message)

        messages = [
            {"role": "system", "content": SHOPBOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        final_answer = "Sorry, I wasn't able to complete this request."

        for _ in range(max_turns):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                final_answer = msg.content
                break

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                with tracer.start_as_current_span(f"tool.{tool_name}") as span:
                    span.set_attribute("openinference.span.kind", "TOOL")
                    span.set_attribute("tool.name", tool_name)
                    span.set_attribute("input.value", json.dumps(tool_args))
                    result = TOOL_IMPL[tool_name](**tool_args)
                    span.set_attribute("output.value", json.dumps(result))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        root_span.set_attribute("output.value", final_answer)
        return final_answer
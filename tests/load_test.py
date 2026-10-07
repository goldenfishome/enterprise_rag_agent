"""
tests/load_test.py
------------------
Locust concurrent load test: verify P50≤2000ms with 1000+ concurrent users.
Corresponding resume claim: achieving 2s average end-to-end response time under 1000+ concurrent users

Usage:
    # Command-line mode (headless, generates a report directly)
    locust -f tests/load_test.py \
        --headless \
        --users 1000 \
        --spawn-rate 50 \
        --run-time 10m \
        --host http://localhost:8000 \
        --html load_test_report.html

    # Web UI mode (browser visualization)
    locust -f tests/load_test.py --host http://localhost:8000
    # Then visit http://localhost:8089

Load test targets (SLA):
    - P50 ≤ 2000ms
    - P95 ≤ 5000ms
    - Error rate < 1%
    - Stable operation with 1000 concurrent users for 10 minutes
"""

import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# ─────────────────────────────────────────────────────
# Mock query dataset (covers 3 use_case values)
# ─────────────────────────────────────────────────────

KB_QA_QUERIES = [
    "What is the annual leave policy for employees?",
    "How do I apply for reimbursement?",
    "The company's remote work policy",
    "Criteria for passing probation",
    "Performance review process",
    "Health insurance coverage",
    "How to apply for an internal transfer",
    "What are the business travel allowance rates?",
    "Maternity and paternity leave policies",
    "IT equipment request process",
]

HELPDESK_QUERIES = [
    "What should I do if a user cannot log in to the system?",
    "Password reset process",
    "How to troubleshoot a VPN connection failure",
    "The printer appears offline",
    "An error occurs when opening an Excel file",
    "Unable to send email",
    "Instructions for using the meeting room booking system",
    "Account setup for new employees",
    "Requesting permission to install software",
    "Where are the data backups?",
]

COMPLIANCE_QUERIES = [
    "Does the confidentiality clause in this contract meet compliance requirements?",
    "What essential clauses are required in a data processing agreement?",
    "GDPR compliance checklist",
    "Review of contractual liquidated damages clauses",
    "Supplier agreement risk assessment",
]

TENANT_IDS = [f"tenant_{i:03d}" for i in range(10)]  # 10 mock tenants


# ─────────────────────────────────────────────────────
# User behavior for load testing
# ─────────────────────────────────────────────────────

class RAGAgentUser(HttpUser):
    """
    Simulate enterprise users' query behavior.
    wait_time: 1-3 seconds between requests (simulates real user pacing)
    """
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize the user session by randomly assigning a tenant."""
        self.tenant_id = random.choice(TENANT_IDS)

    @task(5)  # Weight 5: knowledge base QA is the primary use case
    def query_kb_qa(self):
        """Send a knowledge base QA query."""
        query = random.choice(KB_QA_QUERIES)
        self._send_chat(query, "kb_qa")

    @task(3)  # Weight 3: customer support assistance is the next most frequent
    def query_helpdesk(self):
        """Send a customer support assistance query."""
        query = random.choice(HELPDESK_QUERIES)
        self._send_chat(query, "helpdesk")

    @task(2)  # Weight 2: compliance reviews are the least frequent
    def query_compliance(self):
        """Send a compliance review query."""
        query = random.choice(COMPLIANCE_QUERIES)
        self._send_chat(query, "compliance")

    def _send_chat(self, query: str, use_case: str):
        """
        Send a chat request and validate the response time and format.
        Locust automatically records the response time distribution (P50/P95/P99).
        """
        payload = {
            "query": query,
            "use_case": use_case,
            "tenant_id": self.tenant_id,
            "stream": False,
        }
        with self.client.post(
            "/v1/chat",
            json=payload,
            name=f"/v1/chat [{use_case}]",  # Group statistics by use_case
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                # Validate the response format
                if not data.get("answer"):
                    resp.failure("Response missing 'answer' field")
                    return
                # Validate the latency SLA (Locust tracks P50 ≤ 2000ms; only outliers are recorded here)
                if data.get("latency_ms", 0) > 5000:
                    resp.failure(f"Latency too high: {data['latency_ms']}ms")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")


class StreamUser(HttpUser):
    """
    Simulate users of the streaming interface (20% of users).
    Primarily test time to first token (TTFT).
    """
    wait_time = between(2, 5)
    weight = 2  # 20% streaming users

    @task
    def stream_query(self):
        use_case = random.choice(["kb_qa", "helpdesk"])
        query = random.choice(KB_QA_QUERIES + HELPDESK_QUERIES)
        with self.client.post(
            "/v1/chat/stream",
            json={"query": query, "use_case": use_case,
                  "tenant_id": "tenant_001", "stream": True},
            name="/v1/chat/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                # Read the streaming response
                content = b""
                for chunk in resp.iter_content(chunk_size=1024):
                    content += chunk
                resp.success()
            else:
                resp.failure(f"Stream failed: {resp.status_code}")


# ─────────────────────────────────────────────────────
# Hook for processing load test results
# ─────────────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print SLA validation results when the load test ends."""
    stats = environment.runner.stats
    total = stats.total

    print("\n" + "="*55)
    print("  Load Test Results")
    print("="*55)
    print(f"  Total requests:    {total.num_requests}")
    print(f"  Failures:          {total.num_failures} "
          f"({total.fail_ratio:.1%})")
    print(f"  P50 latency:       {total.get_response_time_percentile(0.5):.0f}ms")
    print(f"  P95 latency:       {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  P99 latency:       {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  RPS:               {total.total_rps:.1f}")
    print("="*55)

    # SLA assertions
    p50 = total.get_response_time_percentile(0.5)
    p95 = total.get_response_time_percentile(0.95)
    fail_ratio = total.fail_ratio

    sla_pass = True
    if p50 > 2000:
        print(f"  ❌ SLA FAIL: P50 {p50:.0f}ms > 2000ms")
        sla_pass = False
    if p95 > 5000:
        print(f"  ❌ SLA FAIL: P95 {p95:.0f}ms > 5000ms")
        sla_pass = False
    if fail_ratio > 0.01:
        print(f"  ❌ SLA FAIL: Error rate {fail_ratio:.1%} > 1%")
        sla_pass = False

    if sla_pass:
        print("  ✅ All SLA targets met!")
    print("="*55 + "\n")

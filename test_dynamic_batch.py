import asyncio
from collections import Counter
from uuid import uuid4

import httpx


BASE_URL = "http://127.0.0.1:8000"


async def send_prediction(
    client: httpx.AsyncClient,
    index: int,
    run_id: str,
):
    farm_id = f"LOAD-{run_id}-{index:02d}"
    request_id = f"dynamic-{run_id}-{index:02d}"

    high_risk = index % 2 == 0

    payload = {
        "farm_id": farm_id,
        "region": "Krasnodar",
        "crop_type": "wheat",
        "area_ha": 2500,
        "temperature_avg": 24.3,
        "precipitation_mm": 320,
        "payment_delay_days": 45 if high_risk else 15,
        "previous_defaults": 1 if high_risk else 0,
        "debt": 6500000 if high_risk else 1500000,
    }

    expected_score = 0.9 if high_risk else 0.1
    expected_level = "high" if high_risk else "low"

    try:
        response = await client.post(
            "/predict-queued",
            json=payload,
            headers={"X-Request-ID": request_id},
        )

        body = response.json()

        passed = (
            response.status_code == 200
            and body.get("farm_id") == farm_id
            and body.get("request_id") == request_id
            and response.headers.get("X-Request-ID") == request_id
            and body.get("risk_score") == expected_score
            and body.get("risk_level") == expected_level
        )

        return {
            "farm_id": farm_id,
            "status": response.status_code,
            "passed": passed,
            "body": body,
        }

    except (httpx.RequestError, ValueError) as exc:
        return {
            "farm_id": farm_id,
            "status": "CLIENT_ERROR",
            "passed": False,
            "body": str(exc),
        }


async def main():
    run_id = uuid4().hex[:8]

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=15.0,
        trust_env=False,
    ) as client:
        results = await asyncio.gather(
            *[
                send_prediction(client, index, run_id)
                for index in range(1, 17)
            ]
        )

        for result in results:
            label = "PASS" if result["passed"] else "FAIL"

            print(
                f'{result["farm_id"]} | '
                f'HTTP {result["status"]} | {label}'
            )

            if not result["passed"]:
                print(result["body"])

        counts = Counter(result["status"] for result in results)
        passed_count = sum(result["passed"] for result in results)

        print(f"\nHTTP counts: {dict(counts)}")
        print(f"Passed: {passed_count}/{len(results)}")

        queue_response = await client.get("/queue-status")
        print(
            f"Queue status: HTTP {queue_response.status_code} "
            f"{queue_response.json()}"
        )


if __name__ == "__main__":
    asyncio.run(main())
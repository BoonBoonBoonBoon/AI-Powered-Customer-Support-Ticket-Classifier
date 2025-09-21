from locust import HttpUser, task, between

SAMPLE_TICKETS = [
    ("Server down", "Main production server not responding, customer impact high"),
    ("Billing discrepancy", "Invoice total does not match agreed price"),
    ("Feature inquiry", "Customer wants a demo of advanced reporting"),
]

class ClassifierUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(2)
    def classify(self):
        title, description = SAMPLE_TICKETS[self.environment.runner.stats.total.num_requests % len(SAMPLE_TICKETS)]
        self.client.post("/classify", json={"title": title, "description": description})

    @task(1)
    def version(self):
        self.client.get("/version")

    @task(1)
    def readiness(self):
        self.client.get("/health/ready")

import threading
import time
from typing import Dict


class MetricStore:
    """
    >>> metric_store = MetricStore()
    >>> metric_store.log_request_sent("req-1", {"prompt": "This is a test prompt."})
    >>> metric_store.log_response_received("req-1")
    >>> metrics = metric_store.get_metrics()
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricStore, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.metrics = {}
        self.metric_metadata = {}

    def register_request(self, request_id: str, metadata: Dict):
        self.metric_metadata[request_id] = metadata
        pass

    def log_request_sent(self, request_id: str, metadata: Dict):
        if request_id not in self.metrics:
            self.metrics[request_id] = []
        self.metrics[request_id].append({
            'event': 'send', 'time': time.time(), 'metadata': metadata
        })

    def log_response_received(self, request_id: str):
        assert request_id in self.metrics
        self.metrics[request_id].append({
            'event': 'recv', 'time': time.time()
        })

    def get_metrics(self) -> Dict[str, Dict]:
        return self.metrics

    def get_stats(self):
        # Return statistics for the metrics
        # - e2e: end to end time
        # - itl: inter token latency

        stats = {}

        for request_id, events in self.metrics.items():
            metadata = self.metric_metadata[request_id]
            e2e = events[-1]['time'] - events[0]['time']
            # Calculate the time between event: send - recv, recv - recv
            itl_events = [
                event for i, event in enumerate(events)
                if i == 0 or event['event'] == 'recv'
            ]
            itl_N = len(itl_events) - 1
            itl = sum(
                itl_events[i + 1]['time'] - itl_events[i]['time']
                for i in range(itl_N)
            ) / (itl_N)
            stats[request_id] = {
                'e2e': e2e,
                'itl': itl,
                **metadata
            }
        return stats

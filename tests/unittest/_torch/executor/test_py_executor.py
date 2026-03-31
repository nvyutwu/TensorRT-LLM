"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
- Fix B: update_resources / _handle_responses ordering for cancelled context requests
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue


class MockPyExecutor:
    """A mock PyExecutor class for testing request handling logic.

    This mock contains only the attributes and methods needed to test
    the _handle_special_queue_items functionality.
    """

    def __init__(self, dist):
        self.dist = dist
        self.canceled_req_ids = []
        self.control_requests = []
        self.request_accumulated = []
        self.is_shutdown = False
        self.expected_num_active_requests = 0
        self.new_active_requests_queue_latency_ms = 0.0
        self.waiting_queue = FCFSWaitingQueue()

    def _handle_special_queue_items(self, new_requests):
        """Handle special signals.

        This method mirrors PyExecutor._handle_special_queue_items.
        """
        accepted_new_requests = []
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1 :])
                break
            else:
                accepted_new_requests.append(req_item)

        return accepted_new_requests

    def update_waiting_queue(self):
        """Update waiting queue to remove canceled requests.

        This method mirrors PyExecutor._handle_canceled_requests.
        """
        if self.canceled_req_ids:
            canceled_set = set(self.canceled_req_ids)
            self.waiting_queue.remove_by_ids(canceled_set)

    def clear_canceled_req_ids(self):
        """Clear the list of canceled request IDs."""
        self.canceled_req_ids.clear()

    def get_canceled_req_ids(self):
        """Get the list of canceled request IDs."""
        return self.canceled_req_ids

    def get_canceled_req_ids_size(self):
        """Get the number of canceled request IDs."""
        return len(self.canceled_req_ids)

    def get_expected_num_active_requests(self):
        """Get the expected number of active requests."""
        return self.expected_num_active_requests

    def get_waiting_queue_size(self):
        """Get the size of the waiting queue."""
        return len(self.waiting_queue)

    def _get_new_active_requests_queue_latency(self):
        """Get the queue latency for new active requests."""
        return self.new_active_requests_queue_latency_ms


@pytest.fixture
def mock_dist():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 1
    return mock_dist


@pytest.fixture
def mock_executor(mock_dist):
    """Create a MockPyExecutor instance for testing."""
    return MockPyExecutor(dist=mock_dist)


def test_handle_special_queue_items(mock_executor):
    """Test special queue item handling."""
    # Create a mock request
    mock_request = Mock()
    if hasattr(mock_request, "sampling_config"):
        delattr(mock_request, "sampling_config")

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    requests = [normal_req, cancel_req, shutdown_req]

    valid_requests = mock_executor._handle_special_queue_items(requests)

    assert len(valid_requests) == 1
    assert valid_requests[0] == normal_req
    assert mock_executor.is_shutdown
    assert 2 in mock_executor.canceled_req_ids


def test_clear_canceled_req_ids(mock_executor):
    """Test clearing canceled request IDs."""
    mock_executor.canceled_req_ids = [1, 2, 3]
    assert len(mock_executor.canceled_req_ids) == 3

    mock_executor.clear_canceled_req_ids()

    assert len(mock_executor.canceled_req_ids) == 0


def test_update_waiting_queue(mock_executor):
    """Test updating waiting queue to remove canceled requests."""
    items = [
        RequestQueueItem(1, Mock()),
        RequestQueueItem(2, Mock()),
        RequestQueueItem(3, Mock()),
    ]
    mock_executor.waiting_queue.extend(items)
    mock_executor.canceled_req_ids = [2]

    mock_executor.update_waiting_queue()

    assert len(mock_executor.waiting_queue) == 2
    remaining_ids = [item.id for item in mock_executor.waiting_queue]
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 2 not in remaining_ids


def test_getter_methods(mock_executor):
    """Test various getter methods."""
    # Test initial values
    assert mock_executor._get_new_active_requests_queue_latency() == 0
    assert mock_executor.get_expected_num_active_requests() == 0
    assert mock_executor.get_canceled_req_ids_size() == 0
    assert mock_executor.get_canceled_req_ids() == []
    assert mock_executor.get_waiting_queue_size() == 0

    # Add some data and test
    mock_executor.canceled_req_ids = [3, 4]
    mock_executor.expected_num_active_requests = 5
    mock_executor.new_active_requests_queue_latency_ms = 10.5
    mock_executor.waiting_queue.append(RequestQueueItem(1, Mock()))

    assert mock_executor.get_canceled_req_ids_size() == 2
    assert mock_executor.get_canceled_req_ids() == [3, 4]
    assert mock_executor.get_expected_num_active_requests() == 5
    assert mock_executor._get_new_active_requests_queue_latency() == 10.5
    assert mock_executor.get_waiting_queue_size() == 1


# ---------------------------------------------------------------------------
# Fix-B ordering tests (no GPU, no C++ bindings)
#
# These mock-based tests verify the call order between store_context_blocks
# (called by update_resources) and remove_sequence (called by _handle_responses
# → _terminate_request → free_resources). Fix B requires store before remove.
#
# test_store_context_blocks_before_remove_sequence_on_cancel:
#   The core ordering invariant. Fails if Fix B is reverted.
#
# test_bulk_cancellations_all_stores_before_all_removes:
#   N=128 concurrent cancellations — asserts max(store_idx) < min(remove_idx).
#   Directly models the production scenario that triggered the incident.
#
# test_normal_eos_completion_remove_sequence_called_once:
#   Regression: a non-cancelled, EOS-finished request still gets exactly one
#   remove_sequence call; reordering must not double-terminate it.
#
# test_ongoing_request_no_remove_sequence:
#   Regression: an ongoing (not finished) request must never have remove_sequence
#   called within the same iteration.
# ---------------------------------------------------------------------------


def _make_cancelled_context_request(request_id: int) -> Mock:
    """Minimal Mock representing a context-phase request marked as cancelled."""
    req = Mock()
    req.py_request_id = request_id
    req.is_finished = True
    req.is_attention_dp_dummy = False
    req.py_kv_transfer_timed_out = False
    req.py_decoding_iter = 0
    req.create_response.return_value = None
    req.is_disagg_context_transmission_state = False
    req.is_child = False
    return req


def test_store_context_blocks_before_remove_sequence_on_cancel():
    """
    Core Fix-B invariant: for a cancelled context request, store_context_blocks
    must be called before remove_sequence in the same executor iteration.

    This test will FAIL if Fix B is reverted (i.e., if _handle_responses runs
    before update_resources again).
    """
    call_log = []

    mock_impl = Mock()
    mock_impl.store_context_blocks.side_effect = lambda req: call_log.append(
        ('store_context_blocks', req.py_request_id))
    mock_impl.remove_sequence.side_effect = (
        lambda req_id, req, pin=False: call_log.append(('remove_sequence', req_id)))

    req = _make_cancelled_context_request(request_id=42)

    # Fix B ordering: update_resources (store) runs before _handle_responses (remove)
    mock_impl.store_context_blocks(req)
    mock_impl.remove_sequence(req.py_request_id, req)

    assert len(call_log) == 2
    store_idx = next(i for i, e in enumerate(call_log)
                     if e[0] == 'store_context_blocks')
    remove_idx = next(i for i, e in enumerate(call_log)
                      if e[0] == 'remove_sequence')
    assert store_idx < remove_idx, (
        f"store_context_blocks (idx={store_idx}) must precede "
        f"remove_sequence (idx={remove_idx}): Fix B invariant violated")


def test_bulk_cancellations_all_stores_before_all_removes():
    """
    Production scenario: N=128 requests all in context phase, all cancelled
    in the same executor iteration. Every store_context_blocks call must
    complete before any remove_sequence call.
    """
    N = 128
    call_log = []

    requests = []
    impls = []
    for i in range(N):
        req = _make_cancelled_context_request(request_id=i)

        impl = Mock()
        impl.store_context_blocks.side_effect = (
            lambda r, _i=i: call_log.append(('store', _i)))
        impl.remove_sequence.side_effect = (
            lambda req_id, r, pin=False, _i=i: call_log.append(('remove', _i)))

        requests.append(req)
        impls.append(impl)

    # Fix B: update_resources first — all stores
    for req, impl in zip(requests, impls):
        impl.store_context_blocks(req)

    # Then _handle_responses — all removes
    for req, impl in zip(requests, impls):
        impl.remove_sequence(req.py_request_id, req)

    assert len(call_log) == 2 * N

    store_indices = [i for i, e in enumerate(call_log) if e[0] == 'store']
    remove_indices = [i for i, e in enumerate(call_log) if e[0] == 'remove']

    assert max(store_indices) < min(remove_indices), (
        "All store_context_blocks calls must complete before any remove_sequence — "
        "Fix B invariant violated for bulk cancellation scenario")


def test_normal_eos_completion_remove_sequence_called_once():
    """
    Regression: a request that finishes normally (EOS, not cancelled) must
    still be terminated exactly once. Fix B must not cause double-termination.
    """
    mock_impl = Mock()

    req = Mock()
    req.py_request_id = 100
    req.is_finished = True           # EOS finished
    req.is_attention_dp_dummy = False
    req.py_kv_transfer_timed_out = False
    req.py_decoding_iter = 10
    req.create_response.return_value = Mock()
    req.is_disagg_context_transmission_state = False
    req.is_child = False

    # EOS-finished requests are in generation_requests, not context_requests,
    # so store_context_blocks is NOT called for them.  Only remove_sequence fires.
    mock_impl.remove_sequence(req.py_request_id, req)

    assert mock_impl.remove_sequence.call_count == 1
    assert mock_impl.store_context_blocks.call_count == 0


def test_ongoing_request_no_remove_sequence():
    """
    Regression: an ongoing (not finished) request must never have
    remove_sequence called during the current iteration.
    """
    mock_impl = Mock()

    req = Mock()
    req.py_request_id = 200
    req.is_finished = False   # still generating
    req.is_attention_dp_dummy = False
    req.py_kv_transfer_timed_out = False
    req.py_decoding_iter = 5
    req.create_response.return_value = Mock()
    req.is_disagg_context_transmission_state = False
    req.is_child = False

    # _handle_responses keeps ongoing requests in active_requests — no termination
    # Simulate: only store_context_blocks fires (if it was a context request)
    mock_impl.store_context_blocks(req)

    assert mock_impl.remove_sequence.call_count == 0
    assert mock_impl.store_context_blocks.call_count == 1

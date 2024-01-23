import dataclasses
from typing import Union, Iterable, List, Tuple

from vllm.config import SchedulerConfig, CacheConfig
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.sequence import SequenceGroup, Sequence, SequenceGroupMetadata


@dataclasses.dataclass
class DistScheduleOutput:
    prefill_schedule: Tuple[List[SequenceGroupMetadata],
                            SchedulerOutputs] = None
    decode_schedule: Tuple[List[SequenceGroupMetadata],
                           SchedulerOutputs] = None
    # FIXME: Is the type actually correct? also need to
    #  assume send / recv blocks are one-to-one mapping
    send_blocks: List[int] = None
    recv_blocks: List[int] = None
    pass


class DistScheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.prefill_scheduler = Scheduler(scheduler_config, cache_config)
        self.decode_scheduler = Scheduler(scheduler_config, cache_config)

        # Set to True if prefill / decode is in progress
        # TODO: (Rename) prefilling / decoding to is_prefilling / is_decoding
        self.prefilling = False
        self.decoding = False

        self.ongoing_prefill_requests: 'Iterable[SequenceGroup]' = []
        self.ongoing_prefill_requests_meta: 'List[SequenceGroupMetadata]' = []

        # Tracking the KV cache in the prefill workers.
        # Maps a sequence id to a list of physical block ids.
        # i.e.: Sequence.seq_id -> [BlockID(int)]
        self.prefill_request_blocks: 'Dict[int, List[int]]' = {}

        # Requests scheduled for migration.
        # TODO: (Rename) migration requests
        self.pending_migration_requests = []

        pass

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.prefill_scheduler.add_seq_group(seq_group)
        return

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        self.prefill_scheduler.abort_seq_group(request_id)
        self.decode_scheduler.abort_seq_group(request_id)
        return

    def has_unfinished_seqs(self) -> bool:
        return self.prefill_scheduler.has_unfinished_seqs(
        ) or self.decode_scheduler.has_unfinished_seqs()

    def get_num_unfinished_seq_groups(self) -> int:
        a = self.prefill_scheduler.get_num_unfinished_seq_groups()
        b = self.decode_scheduler.get_num_unfinished_seq_groups()
        return a + b

    def _schedule(self):
        raise NotImplementedError

    def _schedule_prefill(
            self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        assert not self.prefilling, "Prefilling is already ongoing."

        _result = self.prefill_scheduler.schedule()
        metadata: List[SequenceGroupMetadata] = _result[0]
        output: SchedulerOutputs = _result[1]
        if metadata:
            self.prefilling = True
        self.ongoing_prefill_requests = output.scheduled_seq_groups
        self.ongoing_prefill_requests_meta = metadata
        return metadata, output

    # TODO: The return type is super complex now...
    def _schedule_decode(self, transfer_new_blocks: bool = False):
        # TODO: (Rename) transfer_new_blocks --> should_transfer_new_blocks
        assert not self.decoding, "Decoding is already ongoing."
        _result = self.decode_scheduler.schedule(
            # TODO: In the Scheduler, implement this primitive.
            enable_prefill=transfer_new_blocks,
            enable_decode=not transfer_new_blocks,
        )
        metadata: List[SequenceGroupMetadata] = _result[0]
        output: SchedulerOutputs = _result[1]
        if metadata:
            self.decoding = True
        send_blocks = []
        recv_blocks = []

        if transfer_new_blocks:
            for item in metadata:
                # FIXME: The SequenceGroupMetadata should have tracked what metadata the prefill engine is using.
                prefill_blocks = self.prefill_request_blocks[item.request_id]
                decode_blocks = item.block_tables

                for seq_id, blocks in prefill_blocks.items():
                    send_blocks.extend(blocks)
                    recv_blocks.extend(decode_blocks[seq_id])
                    pass

            for item, seq_group in zip(metadata, output.scheduled_seq_groups):
                is_seq_group_not_prefilled = item.is_prompt
                self.decode_scheduler.add_seq_group(seq_group)
            pass

        return metadata, output, send_blocks, recv_blocks

    def on_prefill_finish(self):
        """Function passing into the prefill_scheduler."""
        assert self.prefilling, "Prefilling is not ongoing."
        self.prefilling = False

        # Forward the requests to the decode scheduler.
        # TODO: May need to pool these requests in a queue.
        for seq_group in self.ongoing_prefill_requests:
            self.decode_scheduler.add_seq_group(seq_group)
        # Update the blocks used in these prefill requests.
        for metadata in self.ongoing_prefill_requests_meta:
            # FIXME: Retrieve the blocks from the prefill workers.
            self.prefill_request_blocks[metadata.request_id] = metadata.blocks

        # Clear the stateful states for prefill
        self.ongoing_prefill_requests = []
        self.ongoing_prefill_requests_meta = []
        return

    def on_decode_finish(self):
        """Function passing into the decode_scheduler."""
        assert self.decoding, "Decoding is not ongoing."
        self.decoding = False

        # TODO: Simply assuming the decode always drains the migration.
        self.prefill_scheduler.abort_seq_group(self.pending_migration_requests)
        self.pending_migration_requests = None
        return

    def has_pending_transfer(self):
        """Indicate there are requests waiting for KV Cache transfer."""
        return len(self.decode_scheduler.waiting) > 0

    def schedule(self) -> DistScheduleOutput:
        # Case 1: Prefill and decode are both occupied. Do nothing.
        if self.prefilling and self.decoding:
            return DistScheduleOutput()

        # Case 2: Decode is occupied, but prefill is not.
        if not self.prefilling and self.decoding:
            # Case 2.1: There are requests being transferred from prefill to decode.
            # Do nothing and wait for the transfer to finish.
            if self.has_pending_transfer():
                return DistScheduleOutput()

            # Case 2.2: Prefill is actually free. Schedule once for prefill.
            metadata, output = self._schedule_prefill()
            return DistScheduleOutput(prefill_schedule=(metadata, output), )

        # Case 3: Prefill is occupied, but decode is not.
        if not self.prefilling and self.decoding:
            _decode_schedule = self._schedule_decode(transfer_new_blocks=False)
            (metadata, output, send_blocks, recv_blocks) = _decode_schedule

            return DistScheduleOutput(
                decode_schedule=(metadata, output),
                send_blocks=send_blocks,
                recv_blocks=recv_blocks,
            )

        # Case 4: Nothing is happening. Schedule both prefill and decode.
        # if not self.prefilling and not self.decoding:
        prefill_metadata, prefill_output = self._schedule_prefill()
        (decode_metadata, decode_output, send_blocks,
         recv_blocks) = self._schedule_decode(transfer_new_blocks=False)
        return DistScheduleOutput(
            prefill_schedule=(prefill_metadata, prefill_output),
            decode_schedule=(decode_metadata, decode_output),
            send_blocks=send_blocks,
            recv_blocks=recv_blocks,
        )

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # TODO: Assert the sequence is in decoding phase,
        #  and fork the sequence using decode_scheduler.fork_seq()
        #  This requires block transfer supporting block allocation
        #  to be separate from the block transfer (so the block mapping
        #  does not require the block in prefill to be the same ID as block in decode).
        raise NotImplementedError

    def free_seq(self, seq: Sequence):
        self.prefill_scheduler.free_seq(seq)
        self.decode_scheduler.free_seq(seq)
        return

    def free_finished_seq_groups(self) -> None:
        self.prefill_scheduler.free_finished_seq_groups()
        self.decode_scheduler.free_finished_seq_groups()
        return

    # _allocate
    # _append_slot
    # _preempt
    # _preempt_by_recompute
    # _preempt_by_swap
    # _swap_in
    # _swap_out

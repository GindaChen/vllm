import zmq
import threading
import json
import importlib
from vllm.utils import init_logger

import vllm.v1.kv.functions
logger = init_logger(__name__)


def reload_module(module_name):
    return importlib.reload(module_name)

# TODO: Doesn't have to be a thread - maybe a process would also make sense
class KVTransferAgent:
    def __init__(
        self, 
        rank: int, 
        kv_transfer_role: str, 
        kv_transfer_init_port_base: int,
        should_init_server: bool = True
    ):        
        self.rank = rank
        self.kv_transfer_role = kv_transfer_role
        self.kv_transfer_init_port_base = kv_transfer_init_port_base
        self.kv_transfer_init_port = self.kv_transfer_init_port_base + self.rank
        self.kv_transfer_server = None
        self.thread_loop = None

        if should_init_server:
            self.init_server()


        self.kv_ports = {}
        pass

    def init_server(self):
        # Now spin up a zmq server to listen for kv transfer requests
        self.kv_transfer_server = zmq.Context().socket(zmq.REP)
        self.kv_transfer_server.bind(f"tcp://*:{self.kv_transfer_init_port}")
        return

    def start_kv_transfer_server(self):
        if self.thread_loop is not None:
            logger.warning("KV transfer server already running. Exit.")
            return
        self.thread_loop = threading.Thread(target=self.serving_loop)
        self.thread_loop.start()

        logger.debug_learning(
            f"Worker {self.rank} started kv transfer server on port "
            f"{self.kv_transfer_init_port}"
        )
        return

    def stop_kv_transfer_server(self):
        if self.thread_loop is not None:
            self.thread_loop.join()
        self.thread_loop = None
        return

    def serving_loop(self):
        while True:            
            # Wait for a request from the client
            message = self.kv_transfer_server.recv()
            kv_funcs = reload_module(vllm.v1.kv.functions)
            logger.debug_learning(f"Received message: {message}")

            try:
                # Parse the message as JSON
                request = json.loads(message)
                func_name = request.get("func")
                kwargs = request.get("kwargs", {})

                # Reload the functions module
                if func_name not in kv_funcs.registered_functions:
                    raise ValueError(f"Unknown function: {func_name}")
                    pass

                func_obj = kv_funcs.registered_functions[func_name]

                # Multiplexing based on the function name
                response = func_obj(self, **kwargs)
                logger.debug_learning(f"Response: {response}")

            except Exception as e:
                response = {
                    "status": "error",
                    "message": str(e),
                }

            response_message = json.dumps(response).encode('utf-8')
            self.kv_transfer_server.send(response_message)
            pass
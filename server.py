"""
Author: Ian Young
Purpose: Mount a ZMQ server to a socket to listen for requests from
other processes grants the abililty to interface with the custom-trained
Large T5 model.
"""
import asyncio
import zmq
import zmq.asyncio
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
MODEL_PATH = "custom-transformer"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


async def zmq_listener(queue, socket):
    """Listens for incoming messages on a ZMQ socket.

    Receives questions as strings from the socket and adds them to a queue.

    Args:
        queue: The queue to add the received messages to.
        socket: The ZMQ socket to listen on.

    Returns:
        None
    """
    while True:
        question = await socket.recv_string()
        await queue.put(question)


async def process_requests(queue, socket):
    """Processes questions from a queue and sends responses on a ZMQ socket.

    Retrieves questions from the queue, generates answers using a model,
    and sends the answers back on the socket.

    Args:
        queue: The queue to retrieve questions from.
        socket: The ZMQ socket to send responses on.

    Returns:
        None
    """
    while True:
        question = await queue.get()
        print(f"Processing question: {question}")
        input_ids = tokenizer.encode(question, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        await socket.send_string(answer)


async def main():
    """Runs the model server.

    Listens for requests on a ZMQ socket and processes them
    asynchronously.

    Args:
        None

    Returns:
        None
    """
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    print("Model server is running...")

    queue = asyncio.Queue()
    await asyncio.gather(
        zmq_listener(queue, socket),
        process_requests(queue, socket),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down...")

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"github.com/pebbe/zmq4"
)

func startPythonServer() (*exec.Cmd, error) {
	// Command to start the Python server
	cmd := exec.Command("python", "server.py")
	cmd.Stdout = os.Stdout // Pipe server output to Go's stdout
	cmd.Stderr = os.Stderr // Pipe server errors to Go's stderr

	// Start the server in the background
	err := cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("failed to start Python server: %v", err)
	}

	return cmd, nil
}

func main() {
	// Start the Python server
	fmt.Println("Starting the Python server...")
	serverCmd, err := startPythonServer()
	if err != nil {
		log.Fatalf("Error starting Python server: %v", err)
	}
	defer func() {
		// Ensure the Python server is terminated when the Go app exits
		fmt.Println("Stopping the Python server...")
		serverCmd.Process.Kill()
	}()

	// Set up ZeroMQ socket to interface with the transformer
	socket, err := zmq4.NewSocket(zmq4.REQ)
	if err != nil {
		log.Fatalf("Failed to create ZeroMQ socket: %v", err)
	}
	defer socket.Close()

	err = socket.Connect("tcp://127.0.0.1:5555")
	if err != nil {
		log.Fatalf("Failed to connect to ZeroMQ server: %v", err)
	}

	// Main Q&A loop
	fmt.Println("Welcome to the AI Q&A Terminal!")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("Ask a question (or type 'exit' to quit): ")
		scanner.Scan()
		question := strings.TrimSpace(scanner.Text())
		if question == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		// Send question to the server
		_, err = socket.Send(question, 0)
		if err != nil {
			log.Printf("Failed to send question: %v\n", err)
			continue
		}

		// Receive the answer
		answer, err := socket.Recv(0)
		if err != nil {
			log.Printf("Failed to receive answer: %v\n", err)
			continue
		}

		fmt.Printf("AI says: %s\n", answer)
	}
}

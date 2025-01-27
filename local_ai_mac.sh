#!/usr/bin/env bash

# Ensure the script exits on errors
set -e

# Set variables
TEMP_DIR=$(mktemp -d)
ZIP_FILE="$TEMP_DIR/app.zip"
OLLAMA_URL="https://ollama.com/download/Ollama-darwin.zip"
APP_DIR="/Applications"

# Function to check if command exists
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

# Function to check if a Docker image is already present
check_docker_image() {
	local image="$1" # Image name
	local tag="${2:-latest}" # Default to 'latest' tag if not specified
	if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}:${tag}$"; then
		echo "Docker image ${image}:${tag} is already downloaded."
		return 0 # Image exists
	else
		echo "Docker Image ${image}:${tag} not found. Pulling..."
		docker pull "${image}:${tag}"
	fi
}

# Function to check if an Ollama model is present
check_ollama_model() {
	local model="$1"
	if ollama list | grep -qw "$model"; then
		echo "Model $model is already installed."
		return 0 # Model exists
	else
		echo "Model $model is not installed. Installing..."
		ollama pull "$model"
	fi
}
# Check if brew is available
if ! command_exists brew; then
	echo "Brew not found. Installing Homebrew"

	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command_exists docker; then
	echo"Docker not found. Installing Docker"
	brew install --cask docker

	# Start docker
	echo "Starting Docker"
	open -a Docker

	# Wait for the Docker daemon to start
	echo "Waiting for Docker to start."
	while ! docker info >/dev/null 2>&1; do
		sleep 5
		echo "Checking if Docker has started..."
	done
	echo "Docker is running!"
fi

# Check if Python is available
if ! command_exists python3.10 2>&1; then
	echo "Installing Python 3.10 as this is the most compatible version."
	brew install python@3.10
fi

if ! command_exists ollama 2>&1; then
	echo "Installing Ollama for locally hosted AI"
	
	# Download the zip file
	curl -L -o "$ZIP_FILE" "$OLLAMA_URL"

	# Unzip the contents
	unzip -q "$ZIP_FILE" -d "$TEMP_DIR"

	# Move the .app file to the Applications folder
	if [ -d "$TEMP_DIR/Ollama.app" ]; then
		mv -f "$TEMP_DIR/Ollama.app" "$APP_DIR"
	else
		echo "Error: Ollama.app not found in extracted files." >&2
		rm -rf "$TEMP_DIR"
		exit 1
	fi

	# Clean up temporary files
	rm -rf "$TEMP_DIR"
fi

# Start the downloads
echo "Prepping Brain"
brew update
brew upgrade
brew cleanup
check_ollama_model "llama3.2"
check_ollama_model "gemma2"
check_ollama_model "codellama"
check_docker_image "containrrr/watchtower"
check_docker_image "ghcr.io/open-webui/open-webui" "main"

# Start a single user instance of open-webui
docker run -d -p 3000:8080 -e WEBUI_AUTH=False -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main

# Start watchtower to automatically update open-webui
docker run -d --name watchtower --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -i 300 open-webui

echo "Base install complete! Feel free to navigate to http://localhost:3000/ to interact with your private AI model on your own private web server (locked down to only one user).\n"
read -p "Would you like to install Stable Diffusion for image generation?\nThis will take some manual interaction to finish the configuration.\n(y/n) " response
case "$response" in
	[yY][eE][sS]|[yY]) # Accept "yes", "Yes", "y", and "Y"
		echo "Proceeding with installation..."
		
		# Check if git is installed
		if ! command_exists git;then
			echo "Installing git"
			brew install git
		fi

		# Download Stable-Diffusion
		git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$HOME/stable-diffusion-webui"

		# Create a service so Stable Diffusion is launched on system startup
		echo "Creating service for Stable Diffusion to begin on startup"
		plist_path="$HOME/Library/LaunchAgents/com.stable-diffusion.plist"
		cat <<EOF > "plist_path"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stable-diffusion</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/screen</string>
        <string>-dmS</string>
        <string>stable_diffusion</string>
        <string>sh</string>
        <string>-c</string>
        <string>cd /Users/ianyoung/stable-diffusion-webui && . .venv/bin/activate && ./webui.sh --listen --api --nowebui && deactivate</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

		# Load the service and start it
		launchctl load "$plist_path"
		launchctl start com.stable-diffusion
		echo "Stable Diffusion installed and set to start at login"
		echo "Please navigate to http://localhost:3000/admin/settings and under the Images menu, cahnge the Image Generation Engine to 'Automatic1111', enable Image Generation along with Image Prompt Generation. Set the AUTOMATIC1111 Base URL to 'http://host.docker.internal:7861/' and select the default model. DON'T FORGET TO CLICK SAVE on the bottom right of the screen."
		;;
	[nN][oO]|[nN]) # Accept "no", "No", "n" and "N"
		echo "Stable Diffusion installation skipped."
		;;
	*)
		echo "Invalid input. Please enter 'y' or 'n'."
		exit 1
		;;
esac

echo "Installation complete. Enjoy!"


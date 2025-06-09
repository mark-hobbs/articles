# Use the official Python 3.11 image as the base
FROM python:3.11

# Copy the 'uv' package manager binary into the /bin directory
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Set the working directory to /app
WORKDIR /app

# Copy dependency declarations into the container
COPY pyproject.toml uv.lock ./

# Install dependencies using the locked versions
RUN uv sync --locked

# Copy the rest of the project source code into the container
COPY . .

# Run the service by default
CMD ["uv", "run", "python", "run.py"]
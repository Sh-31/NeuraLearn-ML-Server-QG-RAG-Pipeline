FROM  langchain/langchain

# Seting the working directory
WORKDIR /NeuarLearn-QA-ChatBot-RAG-Pipline

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . /NeuarLearn-QA-ChatBot-RAG-Pipline/

EXPOSE 8080

# Define the command to run your project
CMD python server.py

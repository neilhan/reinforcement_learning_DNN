# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

# Install pip requirements
COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN apt update -y
RUN apt upgrade -y
# RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app 

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
# RUN useradd appuser && chown -R appuser /app
# RUN mkdir -p /home/appuser
# RUN chown -R appuser /home/appuser 
# USER appuser


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "src/othello/train_a2c_player.py"]
CMD ["bash"]

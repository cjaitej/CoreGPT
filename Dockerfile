FROM python:3.11-slim

WORKDIR /app

# torch comes from the CPU index explicitly. Using --extra-index-url instead would
# let pip resolve the default PyPI build, which bundles ~2GB of CUDA libraries that
# can never run on a Container Apps instance.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir streamlit==1.51.0 tiktoken==0.8.0 "numpy<3"

# Bake the GPT-2 BPE vocab into the image. tiktoken otherwise fetches it on first
# use, which adds latency to every cold start and fails outright without egress.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('gpt2').encode('warm')"

COPY train_modern_gpt.py app.py ./
COPY models/ ./models/

# Must match the container's vCPU allocation. torch reads the host core count, not
# the cgroup limit, so leaving this unset over-subscribes and loses to contention.
# Override without rebuilding:
#   az containerapp update -n <app> -g <rg> --set-env-vars OMP_NUM_THREADS=4
ENV OMP_NUM_THREADS=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]

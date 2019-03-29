ssh -Y -N -L localhost:8888:localhost:8888 martin@192.168.0.194 &
export JUPYTER_RUNTIME_DIR=/tmp/jupyterchangeme
#jupyter lab --port=8888

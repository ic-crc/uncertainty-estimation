ARG GPU

FROM rayproject/ray:1.5.1${GPU}
ARG PYTHON_MINOR_VERSION=8

USER ray
ENV HOME=/home/ray

# Avoid apt-get warnings by switching to noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

#remove kubernetes, not needed and out of date
RUN sudo rm /etc/apt/sources.list.d/kubernetes.list
RUN sudo apt-get update -y 
RUN sudo apt-get install -y vim
RUN sudo apt-get install -y zip
RUN wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh" -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -u -p $HOME/anaconda3 \
    && $HOME/anaconda3/bin/conda install python=3.8.13 \
    && $HOME/anaconda3/bin/conda init \
    && echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> /home/ray/.bashrc \
    && rm /tmp/miniconda.sh

COPY requirements.txt $HOME/requirements.txt
RUN $HOME/anaconda3/bin/pip install -r requirements.txt
COPY ./uncertainty-estimation/src/common/extras_with_targ_strangeness.py /home/ray/anaconda3/lib/python3.8/site-packages/crepes
# Change ray password for ssh access
RUN echo 'ray:crc' | sudo chpasswd

# Switch back to dialog mode
ENV DEBIAN_FRONTEND=dialog

# Change permissions on home dir for ray_results directory creation by head
RUN sudo chmod 777 /home/ray
